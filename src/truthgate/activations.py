"""Small hidden-state diagnostics at the last token of a rendered prompt.

The compared positions are the *last prompt tokens*. With a chat template these
usually belong to the assistant-generation suffix, and are not aligned question
tokens. Different prompt lengths are therefore expected. These diagnostics do
not test factual knowledge or a representation's causal role in the answer.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import torch

if TYPE_CHECKING:
    from truthgate.model import CausalLMRunner


@dataclass(frozen=True)
class HiddenSnapshot:
    """Metadata plus independent CPU vectors; never full activation tensors."""

    layers: list[dict[str, Any]]
    vectors: list[torch.Tensor]
    prompt_length: int
    token_position: int
    token_id: int
    token_text: str


def extract_last_token_states(
    runner: CausalLMRunner, prompt: str
) -> HiddenSnapshot:
    """Extract embedding and block states using a single inference-only forward.

    Hugging Face Qwen3 returns ``num_hidden_layers + 1`` hidden states: the
    embedding output, intermediate block outputs, and the final block output
    *after* the model's final normalization. The final entry must not be treated
    as the unnormalized last-block output. ``logits_to_keep=1`` prevents an
    unnecessary full-prompt vocabulary-sized logits allocation.
    """
    prepared = runner.prepare_prompt(prompt)
    input_ids = prepared.input_ids
    attention_mask = prepared.attention_mask
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("Hidden-state extraction requires a single [1, T] prompt.")
    if input_ids.shape[1] == 0:
        raise ValueError("Cannot extract hidden states from an empty prompt.")
    if attention_mask.shape != input_ids.shape:
        raise ValueError("The attention mask must have the same [1, T] shape as input IDs.")
    if not bool(attention_mask[0, -1].item()):
        raise ValueError("The last prompt position must be an attended, non-padding token.")

    prompt_length = int(input_ids.shape[1])
    token_position = prompt_length - 1
    token_id = int(input_ids[0, token_position].item())
    token_text = runner.tokenizer.decode([token_id], skip_special_tokens=False)
    expected_layers = int(runner.model.config.num_hidden_layers)
    expected_width = int(runner.model.config.hidden_size)
    expected_shape = [1, prompt_length, expected_width]

    layers: list[dict[str, Any]] = []
    vectors: list[torch.Tensor] = []
    with torch.inference_mode():
        outputs = runner.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
            logits_to_keep=1,
        )
        hidden_states = outputs.hidden_states
        if hidden_states is None or len(hidden_states) != expected_layers + 1:
            actual = None if hidden_states is None else len(hidden_states)
            raise ValueError(
                f"Expected {expected_layers + 1} hidden states (embedding + "
                f"{expected_layers} blocks), received {actual}."
            )
        for state_index, state in enumerate(hidden_states):
            shape = list(state.shape)
            if shape != expected_shape:
                raise ValueError(
                    f"Hidden state {state_index} has shape {shape}; "
                    f"expected {expected_shape}."
                )
            # copy=True is essential on CPU too: a view would retain the full
            # [1, T, H] storage even after outputs and hidden_states are released.
            vector = state[0, token_position, :].detach().to(
                device="cpu", dtype=torch.float32, copy=True
            )
            norm = float(torch.linalg.vector_norm(vector).item())
            if not bool(torch.isfinite(vector).all().item()):
                raise ValueError(f"Hidden state {state_index} contains non-finite values.")
            if state_index == 0:
                state_kind = "embedding"
            elif state_index == expected_layers:
                state_kind = "final_normalized"
            else:
                state_kind = "transformer_block"
            layers.append(
                {
                    "state_index": state_index,
                    "layer_index": state_index,
                    "hidden_state_shape": shape,
                    "last_token_norm": norm,
                    "state_kind": state_kind,
                }
            )
            vectors.append(vector)
        # No full-state references escape this function, including the loop's
        # final state reference. Only H-element CPU copies are returned.
        del state, hidden_states, outputs

    return HiddenSnapshot(
        layers=layers,
        vectors=vectors,
        prompt_length=prompt_length,
        token_position=token_position,
        token_id=token_id,
        token_text=token_text,
    )


def compare_hidden_states(
    neutral: HiddenSnapshot,
    biased: HiddenSnapshot,
    *,
    question_id: str,
    option_order: str,
    condition: str,
) -> list[dict[str, Any]]:
    """Return one JSON-serializable row per state for two last-token snapshots.

    Cosine similarity is undefined for a zero vector and is recorded as null
    (``None``), rather than assigning it an arbitrary similarity of zero.
    """
    count = len(neutral.layers)
    if not count or any(
        len(items) != count
        for items in (biased.layers, neutral.vectors, biased.vectors)
    ):
        raise ValueError("Snapshots must contain matching, nonempty states and vectors.")

    rows: list[dict[str, Any]] = []
    for neutral_layer, biased_layer, neutral_vector, biased_vector in zip(
        neutral.layers, biased.layers, neutral.vectors, biased.vectors, strict=True
    ):
        if any(
            neutral_layer[key] != biased_layer[key]
            for key in ("state_index", "layer_index", "state_kind")
        ):
            raise ValueError("Snapshots must have matching hidden-state indices and kinds.")
        if neutral_vector.ndim != 1 or neutral_vector.shape != biased_vector.shape:
            raise ValueError("Compared hidden-state vectors must have matching [H] shapes.")
        # Float64 on the tiny CPU vectors makes the diagnostic numerically stable
        # without retaining or converting the full activation tensors.
        left = neutral_vector.detach().to(device="cpu", dtype=torch.float64)
        right = biased_vector.detach().to(device="cpu", dtype=torch.float64)
        if not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
            raise ValueError("Compared hidden-state vectors must contain finite values.")
        left_norm = float(torch.linalg.vector_norm(left).item())
        right_norm = float(torch.linalg.vector_norm(right).item())
        cosine = None
        if left_norm > 0.0 and right_norm > 0.0:
            cosine = float(torch.dot(left / left_norm, right / right_norm).item())
            cosine = max(-1.0, min(1.0, cosine))
        rows.append(
            {
                "question_id": question_id,
                "option_order": option_order,
                "condition": condition,
                "state_index": neutral_layer["state_index"],
                "layer_index": neutral_layer["layer_index"],
                "state_kind": neutral_layer["state_kind"],
                "neutral_hidden_state_shape": list(neutral_layer["hidden_state_shape"]),
                "biased_hidden_state_shape": list(biased_layer["hidden_state_shape"]),
                "neutral_last_token_norm": left_norm,
                "biased_last_token_norm": right_norm,
                "cosine_similarity": cosine,
                "neutral_prompt_length": neutral.prompt_length,
                "biased_prompt_length": biased.prompt_length,
                "neutral_token_position": neutral.token_position,
                "biased_token_position": biased.token_position,
                "neutral_token_id": neutral.token_id,
                "biased_token_id": biased.token_id,
                "neutral_token_text": neutral.token_text,
                "biased_token_text": biased.token_text,
            }
        )
    return rows


def plot_hidden_comparison(
    rows: Sequence[Mapping[str, Any]], output_path: str | Path
) -> Path:
    """Save an Agg PNG of mean last-prompt-token cosine by condition and state."""
    if not rows:
        raise ValueError("At least one hidden-state comparison row is required.")
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    grouped: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        bucket = grouped[str(row["condition"])][int(row["state_index"])]
        value = row["cosine_similarity"]
        if value is not None:
            value = float(value)
            if not -1.0 <= value <= 1.0:
                raise ValueError("Cosine similarities must be finite numbers in [-1, 1].")
            bucket.append(value)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    try:
        for condition, states in sorted(grouped.items()):
            indices = sorted(states)
            means = [
                sum(states[index]) / len(states[index]) if states[index] else float("nan")
                for index in indices
            ]
            axis.plot(indices, means, marker="o", markersize=3, label=condition)
        axis.set(
            xlabel="Hidden-state index (0 = embeddings; final = normalized output)",
            ylabel="Mean cosine similarity to neutral",
            title="Last prompt token: hidden-state comparison",
            ylim=(-1.05, 1.05),
        )
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(path, format="png", dpi=160)
    finally:
        plt.close(figure)
    return path
