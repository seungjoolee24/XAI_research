"""Exact chat-prefix candidate scoring and unconstrained greedy generation."""

from __future__ import annotations

from dataclasses import dataclass
import os
import random
import re
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def set_seed(seed: int) -> None:
    """Request repeatability; hardware/library changes can still affect numerics."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def select_device_dtype(device: str = "auto", dtype: str = "auto") -> tuple[torch.device, torch.dtype]:
    if device == "auto":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Use a Colab GPU runtime, or explicitly select --device cpu for small-model tests.")
        device = "cuda"
    selected = torch.device(device)
    if selected.type not in {"cuda", "cpu"}:
        raise ValueError("Only cpu and cuda devices are supported in Milestone 1.")
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested but CUDA is unavailable.")
    if dtype == "auto":
        if selected.type == "cpu":
            return selected, torch.float32
        with torch.cuda.device(selected):
            return selected, torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
    choices = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    if dtype not in choices:
        raise ValueError(f"Unsupported dtype: {dtype}")
    if selected.type == "cpu" and dtype != "float32":
        raise ValueError("CPU validation uses float32. Select dtype auto or float32.")
    if selected.type == "cuda" and dtype == "bfloat16":
        with torch.cuda.device(selected):
            if not torch.cuda.is_bf16_supported(including_emulation=False):
                raise ValueError("This GPU does not support BF16; use dtype auto or float16.")
    return selected, choices[dtype]


def parse_answer(text: str) -> str | None:
    """Strict instruction compliance: explanatory/ambiguous outputs are invalid."""
    answer = text.strip()
    return answer if re.fullmatch(r"[AB]", answer) else None


@dataclass
class PreparedPrompt:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    rendered_prompt: str


class CausalLMRunner:
    """A single-device, batch-one causal LM. No training or intervention hooks."""

    def __init__(self, model: Any, tokenizer: Any, device: str | torch.device,
                 max_input_tokens: int = 512, max_new_tokens: int = 8):
        if max_input_tokens < 1 or max_new_tokens < 1:
            raise ValueError("Token limits must be positive.")
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> CausalLMRunner:
        device, dtype = select_device_dtype(config["device"], config["dtype"])
        common = {"revision": config["revision"], "trust_remote_code": False}
        tokenizer = AutoTokenizer.from_pretrained(config["name"], **common)
        # Whole model on one device: no silent CPU/disk offload in GPU runs.
        model = AutoModelForCausalLM.from_pretrained(
            config["name"], **common, dtype=dtype,
            device_map={"": str(device)}, low_cpu_mem_usage=True,
            attn_implementation=config["attention_implementation"],
            use_safetensors=True,
        )
        return cls(model, tokenizer, device, config["max_input_tokens"], config["max_new_tokens"])

    def prepare_prompt(self, prompt: str) -> PreparedPrompt:
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True,
        )
        # Template already contains special tokens. Do not add BOS/EOS twice.
        encoded = self.tokenizer(rendered, add_special_tokens=False, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.device)
        mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(self.device)
        length = input_ids.shape[-1]
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or length < 1:
            raise ValueError("Expected one nonempty prompt.")
        if length > self.max_input_tokens:
            raise ValueError(f"Prompt has {length} tokens, exceeds max_input_tokens={self.max_input_tokens}; no truncation was applied.")
        context_limit = getattr(self.model.config, "max_position_embeddings", None)
        if context_limit and length + self.max_new_tokens > context_limit:
            raise ValueError("Prompt plus generation budget exceeds the model context window.")
        return PreparedPrompt(input_ids, mask, rendered)

    def _candidate_ids(self, prompt: PreparedPrompt, label: str) -> list[int]:
        prefix = prompt.input_ids[0].tolist()
        full = self.tokenizer.encode(prompt.rendered_prompt + label, add_special_tokens=False)
        # Cross-boundary BPE merges would change the conditioning event. Fail
        # visibly instead of measuring a different prompt or wrong position.
        if full[:len(prefix)] != prefix:
            raise ValueError(f"Candidate {label!r} retokenizes the chat-prefix boundary; exact continuation cannot be scored with this tokenizer/template.")
        suffix = full[len(prefix):]
        if not suffix:
            raise ValueError(f"Candidate {label!r} has no continuation tokens.")
        return suffix

    @torch.inference_mode()
    def score_candidates(self, prompt: PreparedPrompt) -> dict[str, Any]:
        token_ids = {label: self._candidate_ids(prompt, label) for label in ("A", "B")}
        output = self.model(input_ids=prompt.input_ids, attention_mask=prompt.attention_mask,
                            use_cache=False, return_dict=True, logits_to_keep=1)
        next_logits = output.logits[0, -1].float()
        next_log_probs = torch.log_softmax(next_logits, dim=-1)
        logits: dict[str, float | None] = {}
        log_probs: dict[str, float] = {}
        for label, ids in token_ids.items():
            logits[label] = float(next_logits[ids[0]].item()) if len(ids) == 1 else None
            if len(ids) == 1:
                log_probs[label] = float(next_log_probs[ids[0]].item())
                continue
            # Teacher force candidate[:-1]. Its last n logits predict all n
            # candidate tokens, starting at the final prompt position.
            suffix = torch.tensor([ids[:-1]], dtype=torch.long, device=self.device)
            teacher_ids = torch.cat([prompt.input_ids, suffix], dim=-1)
            context_limit = getattr(self.model.config, "max_position_embeddings", None)
            if context_limit and teacher_ids.shape[-1] > context_limit:
                raise ValueError("Candidate exceeds the model context window.")
            teacher_mask = torch.cat([prompt.attention_mask, torch.ones_like(suffix)], dim=-1)
            teacher = self.model(input_ids=teacher_ids, attention_mask=teacher_mask,
                                 use_cache=False, return_dict=True, logits_to_keep=len(ids))
            steps = torch.log_softmax(teacher.logits[0, -len(ids):].float(), dim=-1)
            targets = torch.tensor(ids, device=self.device)
            log_probs[label] = float(steps.gather(1, targets[:, None]).sum().item())
            del teacher, steps
        single = all(len(ids) == 1 for ids in token_ids.values())
        scores = {label: logits[label] if single else log_probs[label] for label in ("A", "B")}
        score_values = torch.tensor([scores["A"], scores["B"]], dtype=torch.float64)
        if not torch.isfinite(score_values).all():
            raise ValueError("Non-finite candidate scores; check model/dtype.")
        probabilities = torch.softmax(score_values, dim=0).tolist()
        margin = float(scores["A"] - scores["B"])
        result = {
            "score_type": "single_token_logit" if single else "sequence_log_probability",
            "candidate_token_ids": token_ids,
            "candidate_token_counts": {label: len(ids) for label, ids in token_ids.items()},
            "candidates_are_single_tokens": single,
            "logit_a": logits["A"], "logit_b": logits["B"],
            "logit_margin": float(logits["A"] - logits["B"]) if single else None,
            "candidate_logprob_a": log_probs["A"], "candidate_logprob_b": log_probs["B"],
            "score_a": scores["A"], "score_b": scores["B"], "score_margin": margin,
            "probability_a": probabilities[0], "probability_b": probabilities[1],
            "candidate_answer": "A" if margin > 0 else ("B" if margin < 0 else None),
        }
        return result

    @torch.inference_mode()
    def generate(self, prompt: PreparedPrompt) -> dict[str, Any]:
        original = getattr(self.model, "generation_config", None)
        eos = getattr(original, "eos_token_id", None)
        if eos is None:
            eos = self.tokenizer.eos_token_id
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = eos[0] if isinstance(eos, list) else eos
        # Fresh config prevents checkpoint sampling/repetition defaults from
        # altering the unconstrained greedy baseline.
        generation = GenerationConfig(
            do_sample=False, num_beams=1, max_new_tokens=self.max_new_tokens,
            use_cache=True, eos_token_id=eos, pad_token_id=pad,
            bos_token_id=getattr(self.tokenizer, "bos_token_id", None),
        )
        output = self.model.generate(input_ids=prompt.input_ids,
                                     attention_mask=prompt.attention_mask,
                                     generation_config=generation)
        ids = output[0, prompt.input_ids.shape[-1]:].tolist()
        text = self.tokenizer.decode(ids, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        return {"generated_text": text, "generated_token_ids": ids,
                "predicted_answer": parse_answer(text),
                "generation_hit_token_limit": len(ids) >= self.max_new_tokens and (not ids or ids[-1] not in eos_ids)}

    def evaluate_prompt(self, prompt: str) -> dict[str, Any]:
        prepared = self.prepare_prompt(prompt)
        return {"prompt": prompt, "rendered_prompt": prepared.rendered_prompt,
                "prompt_token_count": prepared.input_ids.shape[-1],
                "answer_start_token_position": prepared.input_ids.shape[-1],
                **self.score_candidates(prepared), **self.generate(prepared)}
