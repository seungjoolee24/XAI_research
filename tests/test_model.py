from types import SimpleNamespace
from contextlib import nullcontext

import pytest
import torch

from truthgate.model import CausalLMRunner, parse_answer, select_device_dtype


class CharacterTokenizer:
    eos_token_id = 127
    pad_token_id = 0
    bos_token_id = None
    chat_template = "test character template"
    init_kwargs = {}

    def __init__(self, multi=False, merge=False):
        self.multi = multi
        self.merge = merge

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True}
        return "user:" + messages[0]["content"] + "\nassistant:\n"

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        ids = []
        for char in text:
            ids.extend([65, 67] if self.multi and char == "A" else [ord(char)])
        if self.merge and text.endswith("\nA"):
            ids[-2:] = [90]
        return ids

    def __call__(self, text, **kwargs):
        ids = torch.tensor([self.encode(text)], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def decode(self, ids, skip_special_tokens=False, **kwargs):
        return "".join(chr(i) for i in ids if not skip_special_tokens or i not in (0, 127))


class FixedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(max_position_embeddings=1024, num_hidden_layers=2, hidden_size=8)
        self.generation_config = SimpleNamespace(eos_token_id=127, repetition_penalty=2.0)

    def forward(self, input_ids, attention_mask, use_cache, return_dict, logits_to_keep):
        assert not self.training and torch.is_inference_mode_enabled()
        assert not use_cache and return_dict
        logits = torch.zeros((*input_ids.shape, 128), device=input_ids.device)
        logits[..., 65] = 2.0
        logits[..., 66] = 1.0
        logits[..., 67] = (input_ids == 65).float() * 3
        return SimpleNamespace(logits=logits[:, -logits_to_keep:])

    def generate(self, input_ids, attention_mask, generation_config):
        assert torch.is_inference_mode_enabled() and not self.training
        assert not generation_config.do_sample and generation_config.num_beams == 1
        assert generation_config.repetition_penalty == 1.0
        return torch.cat([input_ids, torch.tensor([[66, 127]])], dim=1)


def runner(**kwargs):
    return CausalLMRunner(FixedModel(), CharacterTokenizer(**kwargs), "cpu")


@pytest.mark.parametrize("text,expected", [(" A\n", "A"), ("B", "B"), ("A.", None), ("A or B", None), ("I think A", None), ("", None)])
def test_strict_generation_parse(text, expected):
    assert parse_answer(text) == expected


def test_same_chat_prefix_scoring_and_independent_greedy_answer():
    result = runner().evaluate_prompt("test")
    assert result["rendered_prompt"].endswith("assistant:\n")
    assert result["logit_a"] == 2 and result["logit_b"] == 1
    assert result["logit_margin"] == 1
    assert result["candidate_answer"] == "A"
    assert result["predicted_answer"] == "B"
    assert result["probability_a"] + result["probability_b"] == pytest.approx(1)
    assert result["probability_a"] == pytest.approx(torch.sigmoid(torch.tensor(1.)).item())
    assert result["answer_start_token_position"] == result["prompt_token_count"]
    assert result["generation_hit_token_limit"] is False


def test_multitoken_teacher_forcing_uses_all_tokens_and_correct_alignment():
    model_runner = runner(multi=True)
    prepared = model_runner.prepare_prompt("test")
    result = model_runner.score_candidates(prepared)
    baseline = torch.zeros(128)
    baseline[65], baseline[66] = 2, 1
    after_a = baseline.clone()
    after_a[67] = 3
    expected = baseline.log_softmax(0)[65] + after_a.log_softmax(0)[67]
    assert result["candidate_token_ids"]["A"] == [65, 67]
    assert result["candidate_logprob_a"] == pytest.approx(expected.item())
    assert result["candidate_logprob_b"] == pytest.approx(baseline.log_softmax(0)[66].item())
    assert result["score_type"] == "sequence_log_probability"
    assert result["logit_a"] is None and result["logit_margin"] is None
    assert result["logit_b"] == 1
    assert result["score_a"] == result["candidate_logprob_a"]


def test_boundary_merge_rejected_instead_of_misaligned_score():
    with pytest.raises(ValueError, match="retokenizes"):
        runner(merge=True).evaluate_prompt("test")


def test_overlong_prompt_rejected_without_truncation():
    model_runner = runner()
    model_runner.max_input_tokens = 3
    with pytest.raises(ValueError, match="no truncation"):
        model_runner.prepare_prompt("test")


def test_explicit_cpu_float32_and_no_silent_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert select_device_dtype("cpu") == (torch.device("cpu"), torch.float32)
    with pytest.raises(RuntimeError, match="CUDA"):
        select_device_dtype()
    with pytest.raises(ValueError, match="float32"):
        select_device_dtype("cpu", "float16")


def test_score_tie_has_no_candidate_winner():
    model_runner = runner()
    original = model_runner.model.forward

    def tied(**kwargs):
        result = original(**kwargs)
        result.logits[..., 66] = result.logits[..., 65]
        return result

    model_runner.model.forward = tied
    result = model_runner.score_candidates(model_runner.prepare_prompt("test"))
    assert result["candidate_answer"] is None
    assert result["probability_a"] == result["probability_b"] == 0.5


@pytest.mark.parametrize("native_bf16", [True, False])
def test_cuda_auto_precision_requires_native_bf16_support(monkeypatch, native_bf16):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())

    def supported(*, including_emulation):
        assert including_emulation is False
        return native_bf16

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", supported)
    assert select_device_dtype("cuda")[1] == (torch.bfloat16 if native_bf16 else torch.float16)
    if not native_bf16:
        with pytest.raises(ValueError, match="BF16"):
            select_device_dtype("cuda", "bfloat16")
