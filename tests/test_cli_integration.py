"""Offline CLI integration with a random tiny Qwen3, never research results."""

import json
from pathlib import Path
import os
import subprocess
import sys

import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]


def assert_reports(output):
    from pypdf import PdfReader

    for name in ("pre_experiment.pdf", "post_experiment.pdf"):
        path = output / name
        assert path.read_bytes().startswith(b"%PDF")
        reader = PdfReader(path)
        assert len(reader.pages) >= 1
        assert any(page.extract_text().strip() for page in reader.pages)


@pytest.fixture(scope="module")
def tiny_model_config(tmp_path_factory):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    directory = tmp_path_factory.mktemp("tiny-qwen3-cli")
    model_path = directory / "model"
    vocab = {"<unk>": 0, "<pad>": 1, "<eos>": 2, "A": 3, "B": 4, "<user>": 5, "<assistant>": 6}
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", pad_token="<pad>", eos_token="<eos>")
    tokenizer.chat_template = "{{ '<user> ' + messages[0]['content'] + ' <assistant> ' }}"
    tokenizer.save_pretrained(model_path)
    torch.manual_seed(0)
    config = Qwen3Config(vocab_size=len(vocab), hidden_size=8, intermediate_size=16,
                         num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                         head_dim=4, max_position_embeddings=512,
                         eos_token_id=2, pad_token_id=1, bos_token_id=None)
    model = Qwen3ForCausalLM(config)
    model.save_pretrained(model_path)
    run_config = yaml.safe_load((ROOT / "configs/baseline.yaml").read_text(encoding="utf-8"))
    run_config["model"].update(name=str(model_path), device="cpu", max_new_tokens=3)
    run_config["dataset"]["path"] = str(ROOT / "data/baseline_questions.json")
    run_config["output"]["directory"] = str(directory / "outputs")
    config_path = directory / "baseline.yaml"
    config_path.write_text(yaml.safe_dump(run_config), encoding="utf-8")
    return directory, config_path


def run_cli(script, config_path, output, *extra):
    environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                   "MPLCONFIGDIR": str(output.parent / "mplcache")}
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / script), "--config", str(config_path),
                             "--output-dir", str(output), *extra], cwd=output.parent,
                            env=environment, text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_smoke_cli_loads_saved_qwen_offline(tiny_model_config):
    directory, config_path = tiny_model_config
    output = directory / "smoke"
    run_cli("smoke_inference.py", config_path, output)
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert result["candidates_are_single_tokens"] is True
    assert result["candidate_token_ids"] == {"A": [3], "B": [4]}
    assert result["probability_a"] + result["probability_b"] == pytest.approx(1)
    metadata = json.loads((output / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["device"] == "cpu"
    assert metadata["effective_config"]["seed"] == 42
    assert_reports(output)


def test_baseline_cli_three_conditions_both_orders_and_repeatability(tiny_model_config):
    directory, config_path = tiny_model_config
    outputs = [directory / "baseline-first", directory / "baseline-repeat"]
    for output in outputs:
        run_cli("run_baseline.py", config_path, output, "--limit", "1", "--include-swapped")
    contents = [(output / "results.jsonl").read_text(encoding="utf-8") for output in outputs]
    assert contents[0] == contents[1]
    rows = [json.loads(line) for line in contents[0].splitlines()]
    assert len(rows) == 6
    assert {row["condition"] for row in rows} == {"neutral", "false_opinion", "correct_opinion"}
    assert {row["option_order"] for row in rows} == {"original", "swapped"}
    for row in rows:
        assert row["is_correct"] == (row["predicted_answer"] == row["ground_truth"])
        assert row["truth_score_margin"] == pytest.approx(row["score_margin"] * (1 if row["ground_truth"] == "A" else -1))
    summary = json.loads((outputs[0] / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_rows"] == 6
    # Even when no flips occur, CSV still has the full result header.
    assert "neutral_answer" in (outputs[0] / "flip_cases.csv").read_text(encoding="utf-8").splitlines()[0]
    for output in outputs:
        assert_reports(output)


def test_hidden_cli_writes_small_summaries_and_real_png(tiny_model_config):
    directory, config_path = tiny_model_config
    output = directory / "hidden"
    run_cli("inspect_hidden_states.py", config_path, output, "--limit", "1")
    rows = [json.loads(line) for line in (output / "hidden_states.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 6  # Two conditions x (embedding + two blocks).
    assert {row["state_index"] for row in rows} == {0, 1, 2}
    assert all(row["neutral_hidden_state_shape"][-1] == 8 for row in rows)
    assert (output / "hidden_comparison.png").read_bytes().startswith(b"\x89PNG")
    assert not list(output.glob("*.pt"))
    assert_reports(output)


def test_failed_model_load_still_leaves_pre_and_failure_post_reports(tiny_model_config):
    directory, config_path = tiny_model_config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["model"]["name"] = str(directory / "model-that-does-not-exist")
    failing_config = directory / "failure-config.yaml"
    failing_config.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = directory / "failure"
    environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    process = subprocess.run([sys.executable, str(ROOT / "scripts/smoke_inference.py"),
                              "--config", str(failing_config), "--output-dir", str(output)],
                             cwd=directory, env=environment, text=True, capture_output=True, timeout=120)
    assert process.returncode != 0
    assert_reports(output)
    failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
    assert failure["status"] == "failed"
    assert not (output / "result.json").exists()


def test_notebook_schema_and_code_are_clean():
    import nbformat

    notebook = nbformat.read(ROOT / "notebooks/colab_runner.ipynb", as_version=4)
    nbformat.validate(notebook)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            assert cell.execution_count is None and not cell.outputs
            compile(cell.source, f"colab cell {index}", "exec")
