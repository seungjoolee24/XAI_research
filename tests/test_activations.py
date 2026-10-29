"""CPU-only tests; no pretrained model or tokenizer downloads are required."""

from types import SimpleNamespace

import pytest
import torch

from truthgate.activations import (
    HiddenSnapshot,
    compare_hidden_states,
    extract_last_token_states,
    plot_hidden_comparison,
)


class FakeModel:
    def __init__(self, *, count=3, wrong_shape=False, zero=False):
        self.config = SimpleNamespace(num_hidden_layers=2, hidden_size=4)
        self.count = count
        self.wrong_shape = wrong_shape
        self.zero = zero
        self.states = []
        self.forward_kwargs = None

    def __call__(self, **kwargs):
        assert torch.is_inference_mode_enabled()
        self.forward_kwargs = kwargs
        length = kwargs["input_ids"].shape[1]
        width = 3 if self.wrong_shape else self.config.hidden_size
        self.states = [
            (
                torch.zeros(1, length, width)
                if self.zero
                else torch.arange(length * width, dtype=torch.float32).reshape(
                    1, length, width
                ) + index
            )
            for index in range(self.count)
        ]
        return SimpleNamespace(hidden_states=tuple(self.states))


class FakeRunner:
    def __init__(self, model=None):
        self.model = model or FakeModel()
        self.tokenizer = SimpleNamespace(
            decode=lambda ids, skip_special_tokens: f"<token-{ids[0]}>"
        )
        self.device = torch.device("cpu")
        self.max_input_tokens = 64

    def prepare_prompt(self, prompt):
        ids = torch.arange(1, len(prompt.split()) + 1).unsqueeze(0)
        return SimpleNamespace(
            input_ids=ids, attention_mask=torch.ones_like(ids), rendered_prompt=prompt
        )


def test_extract_copies_only_last_token_cpu_vectors():
    runner = FakeRunner()
    snapshot = extract_last_token_states(runner, "three token prompt")

    assert len(snapshot.layers) == 3
    assert len(snapshot.vectors) == 3
    assert snapshot.prompt_length == 3
    assert snapshot.token_position == 2
    assert snapshot.token_id == 3
    assert snapshot.token_text == "<token-3>"
    assert [item["state_kind"] for item in snapshot.layers] == [
        "embedding", "transformer_block", "final_normalized"
    ]
    assert [item["layer_index"] for item in snapshot.layers] == [0, 1, 2]
    for index, (metadata, vector) in enumerate(zip(snapshot.layers, snapshot.vectors)):
        assert metadata["hidden_state_shape"] == [1, 3, 4]
        assert metadata["last_token_norm"] == pytest.approx(vector.norm().item())
        assert vector.device.type == "cpu"
        assert vector.dtype == torch.float32
        assert vector.shape == (4,)
        assert vector.numel() == vector.untyped_storage().nbytes() // vector.element_size()
        assert not vector.requires_grad
        assert vector.data_ptr() != runner.model.states[index][0, -1, :].data_ptr()
        torch.testing.assert_close(vector, runner.model.states[index][0, -1, :])
    assert runner.model.forward_kwargs["output_hidden_states"] is True
    assert runner.model.forward_kwargs["use_cache"] is False
    assert runner.model.forward_kwargs["return_dict"] is True
    assert runner.model.forward_kwargs["logits_to_keep"] == 1


@pytest.mark.parametrize(
    "model, message",
    [
        (FakeModel(count=2), "Expected 3 hidden states"),
        (FakeModel(wrong_shape=True), "expected \\[1, 2, 4\\]"),
    ],
)
def test_extraction_rejects_unexpected_model_layout(model, message):
    with pytest.raises(ValueError, match=message):
        extract_last_token_states(FakeRunner(model), "two tokens")


def test_extraction_rejects_empty_prompt():
    with pytest.raises(ValueError, match="empty prompt"):
        extract_last_token_states(FakeRunner(), "")


def test_comparison_preserves_different_prompt_positions_and_shapes():
    runner = FakeRunner()
    neutral = extract_last_token_states(runner, "question here")
    biased = extract_last_token_states(runner, "I think A question here")
    rows = compare_hidden_states(
        neutral, biased, question_id="q01", option_order="original", condition="false_opinion"
    )
    assert len(rows) == 3
    for row in rows:
        assert row["question_id"] == "q01"
        assert row["condition"] == "false_opinion"
        assert row["option_order"] == "original"
        assert row["neutral_hidden_state_shape"] == [1, 2, 4]
        assert row["biased_hidden_state_shape"] == [1, 5, 4]
        assert row["neutral_token_position"] == 1
        assert row["biased_token_position"] == 4
        assert row["neutral_token_id"] == 2
        assert row["biased_token_text"] == "<token-5>"
        assert -1.0 <= row["cosine_similarity"] <= 1.0


def test_identical_snapshots_have_cosine_one():
    snapshot = extract_last_token_states(FakeRunner(), "one two")
    rows = compare_hidden_states(
        snapshot, snapshot, question_id="q", option_order="original", condition="correct_opinion"
    )
    assert all(row["cosine_similarity"] == pytest.approx(1.0) for row in rows)


def test_zero_vectors_have_undefined_cosine():
    snapshot = extract_last_token_states(FakeRunner(FakeModel(zero=True)), "one two")
    rows = compare_hidden_states(
        snapshot, snapshot, question_id="q", option_order="original", condition="false_opinion"
    )
    assert all(row["cosine_similarity"] is None for row in rows)
    assert all(row["neutral_last_token_norm"] == 0 for row in rows)


def test_comparison_rejects_mismatched_snapshot_lengths():
    neutral = extract_last_token_states(FakeRunner(), "one two")
    broken = HiddenSnapshot(
        layers=neutral.layers[:-1], vectors=neutral.vectors[:-1], prompt_length=2,
        token_position=1, token_id=2, token_text="<token-2>",
    )
    with pytest.raises(ValueError, match="matching, nonempty"):
        compare_hidden_states(
            neutral, broken, question_id="q", option_order="original", condition="false_opinion"
        )


def test_plot_saves_png_without_interactive_backend(tmp_path):
    snapshot = extract_last_token_states(FakeRunner(), "one two")
    rows = compare_hidden_states(
        snapshot, snapshot, question_id="q", option_order="original", condition="false_opinion"
    )
    path = plot_hidden_comparison(rows, tmp_path / "nested" / "hidden.png")
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_plot_rejects_empty_rows(tmp_path):
    with pytest.raises(ValueError, match="At least one"):
        plot_hidden_comparison([], tmp_path / "hidden.png")


def test_real_tiny_qwen3_hidden_state_api_offline():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    config = Qwen3Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=32,
    )
    model = Qwen3ForCausalLM(config).eval()
    snapshot = extract_last_token_states(FakeRunner(model), "one two three")

    assert len(snapshot.layers) == config.num_hidden_layers + 1
    assert all(layer["hidden_state_shape"] == [1, 3, 8] for layer in snapshot.layers)
    assert all(vector.shape == (8,) for vector in snapshot.vectors)
    assert snapshot.layers[-1]["state_kind"] == "final_normalized"
