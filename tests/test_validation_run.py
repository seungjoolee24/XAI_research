"""Offline checkpoint and frozen-protocol checks; no research model loading."""

import json
from pathlib import Path

import pytest

from truthgate.experiment import load_config, write_json
from truthgate.validation_run import (
    collect_validation, execute_validation, protocol_identity,
    read_partial_checkpoint, runtime_identity, verify_protocol,
)


ROOT = Path(__file__).resolve().parents[1]


class FixtureRunner:
    def __init__(self, stop_after=None):
        self.calls = 0
        self.stop_after = stop_after

    def prepare_prompt(self, prompt):
        assert "Answer only A or B." in prompt

    def evaluate_prompt(self, prompt):
        if self.stop_after is not None and self.calls == self.stop_after:
            raise RuntimeError("fixture interruption")
        self.calls += 1
        return {"prompt": prompt, "generated_text": "A", "predicted_answer": "A",
                "candidate_answer": "A", "score_margin": 2., "logit_margin": 2.}


@pytest.fixture
def config():
    value = load_config(ROOT / "configs/baseline.yaml")
    value["experiment"]["limit"] = 1
    value["dataset"]["include_swapped"] = True
    return value


def test_interrupted_checkpoint_resumes_only_remaining_prompts(config, tmp_path):
    first = FixtureRunner(stop_after=2)
    with pytest.raises(RuntimeError, match="fixture interruption"):
        collect_validation(first, config, tmp_path)
    assert len((tmp_path / "checkpoint.jsonl").read_text().splitlines()) == 2
    second = FixtureRunner()
    rows = collect_validation(second, config, tmp_path, resume=True)
    assert second.calls == 4 and len(rows) == 6
    assert len({(row["question_id"], row["option_order"], row["condition"]) for row in rows}) == 6
    assert rows[0]["truth_score_margin"] == 2.
    assert rows[3]["truth_score_margin"] == -2.
    assert collect_validation(FixtureRunner(stop_after=0), config, tmp_path, resume=True) == rows


def test_existing_checkpoint_requires_explicit_resume(config, tmp_path):
    collect_validation(FixtureRunner(), config, tmp_path)
    with pytest.raises(FileExistsError, match="resume"):
        collect_validation(FixtureRunner(), config, tmp_path)


@pytest.mark.parametrize("corruption", ["duplicate", "prompt", "option", "unknown"])
def test_corrupted_checkpoint_rejected_before_inference(config, tmp_path, corruption):
    collect_validation(FixtureRunner(), config, tmp_path)
    path = tmp_path / "checkpoint.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if corruption == "duplicate":
        rows.append(rows[0])
    elif corruption == "prompt":
        rows[0]["prompt"] = "altered"
    elif corruption == "option":
        rows[0]["option_a"] = "altered"
    else:
        rows[0]["question_id"] = "unknown"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="Checkpoint"):
        collect_validation(FixtureRunner(stop_after=0), config, tmp_path, resume=True)


def test_protocol_rejects_dataset_and_pdf_changes(config, tmp_path):
    import hashlib
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    pdf = tmp_path / "validation_pre_experiment.pdf"
    pdf.write_bytes(b"frozen-pdf-fixture")
    write_json(tmp_path / "execution_protocol.json", {"identity": protocol_identity(config, manifest),
               "pre_pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest()})
    verify_protocol(config, manifest, tmp_path)
    altered = {**config, "seed": 17}
    with pytest.raises(ValueError, match="changed after"):
        verify_protocol(altered, manifest, tmp_path)
    pdf.write_bytes(b"changed")
    with pytest.raises(ValueError, match="PDF changed"):
        verify_protocol(config, manifest, tmp_path)


def test_expanded_notebook_clean_schema_and_compilable_cells():
    import nbformat
    notebook = nbformat.read(ROOT / "notebooks/colab_expanded_validation.ipynb", as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is None and not cell.outputs
            compile(cell.source, "expanded notebook fixture validation", "exec")


def test_completed_run_refuses_before_model_loading_or_file_mutation(config, tmp_path, monkeypatch):
    """Accidental re-execution must preserve the completed scientific evidence."""
    import truthgate.model as model_module
    import truthgate.validation_run as execution

    monkeypatch.setattr(execution, "verify_protocol", lambda *args: {})

    def forbidden_load(*args, **kwargs):
        pytest.fail("A completed run must be refused before model loading")

    monkeypatch.setattr(model_module.CausalLMRunner, "from_config", forbidden_load)
    for filename, payload in {
        "completion.json": b'{"status":"completed"}\n',
        "run_metadata.json": b'{"status":"completed","original":true}\n',
        "validation_post_experiment.pdf": b"original verified report fixture",
        "checkpoint.jsonl": b'{"original":"checkpoint"}\n',
    }.items():
        (tmp_path / filename).write_bytes(payload)
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    with pytest.raises(FileExistsError, match="complete"):
        execute_validation(config, tmp_path / "manifest.json", tmp_path,
                           pre_report_delivered=True)
    after = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert after == before
    assert not (tmp_path / "attempts").exists()


def test_existing_unresumed_checkpoint_refuses_before_any_writes(config, tmp_path, monkeypatch):
    import truthgate.validation_run as execution

    monkeypatch.setattr(execution, "verify_protocol", lambda *args: {})
    checkpoint = tmp_path / "checkpoint.jsonl"
    checkpoint.write_bytes(b"preserved checkpoint fixture\n")
    with pytest.raises(FileExistsError, match="resume"):
        execute_validation(config, tmp_path / "manifest.json", tmp_path,
                           pre_report_delivered=True, runner=object())
    assert list(tmp_path.iterdir()) == [checkpoint]
    assert checkpoint.read_bytes() == b"preserved checkpoint fixture\n"


def _runtime_metadata_fixture():
    return {
        "model_name": "offline-fixture", "model_revision": "model-commit",
        "tokenizer_revision": "tokenizer-commit", "chat_template_sha256": "template-digest",
        "dataset_sha256": "dataset-digest", "device": "cuda:0", "dtype": "torch.float16",
        "versions": {"torch": "fixture-v1"},
        "installed_distributions": {"torch": "fixture-v1", "numpy": "fixture-v1"},
        "inference_settings": {"do_sample": False, "max_new_tokens": 8},
        "runtime": {"platform": "fixture-linux", "cuda_runtime": "fixture-cuda",
                    "gpu_name": "fixture-T4", "cudnn_version": 123,
                    "deterministic_algorithms": True},
        "created_at_utc": "first-attempt", "execution_started_at_utc": "first-attempt",
    }


@pytest.mark.parametrize("field,replacement", [
    ("model_revision", "different-model"),
    ("tokenizer_revision", "different-tokenizer"),
    ("chat_template_sha256", "different-template"),
    ("dataset_sha256", "different-data"),
    ("device", "cpu"),
    ("dtype", "torch.bfloat16"),
    ("versions", {"torch": "fixture-v2"}),
    ("installed_distributions", {"torch": "fixture-v2"}),
    ("inference_settings", {"do_sample": True, "max_new_tokens": 8}),
])
def test_resume_runtime_fingerprint_detects_numerically_relevant_changes(field, replacement):
    original = _runtime_metadata_fixture()
    altered = {**original, field: replacement}
    assert runtime_identity(original) != runtime_identity(altered)


@pytest.mark.parametrize("field,replacement", [
    ("gpu_name", "fixture-A100"),
    ("cuda_runtime", "different-cuda"),
    ("cudnn_version", 456),
    ("deterministic_algorithms", False),
])
def test_resume_runtime_fingerprint_detects_gpu_backend_changes(field, replacement):
    original = _runtime_metadata_fixture()
    altered = {**original, "runtime": {**original["runtime"], field: replacement}}
    assert runtime_identity(original) != runtime_identity(altered)


def test_resume_runtime_fingerprint_ignores_attempt_timestamps():
    original = _runtime_metadata_fixture()
    altered = {**original, "created_at_utc": "later-attempt",
               "execution_started_at_utc": "later-attempt", "attempt_directory": "attempts/002"}
    assert runtime_identity(original) == runtime_identity(altered)


def test_partial_checkpoint_reports_intact_prefix_without_repairing_corruption(tmp_path):
    path = tmp_path / "checkpoint.jsonl"
    first = {"question_id": "fixture-one", "condition": "neutral"}
    second = {"question_id": "fixture-two", "condition": "neutral"}
    payload = (json.dumps(first) + "\n" + '{"question_id":' + "\n" + json.dumps(second) + "\n").encode()
    path.write_bytes(payload)
    rows, error = read_partial_checkpoint(path)
    assert rows == [first]
    assert "line 2" in error
    assert path.read_bytes() == payload


def test_partial_checkpoint_rejects_nonobject_row_and_handles_missing_file(tmp_path):
    assert read_partial_checkpoint(tmp_path / "missing.jsonl") == ([], None)
    path = tmp_path / "checkpoint.jsonl"
    path.write_text('{"question_id":"fixture"}\n[]\n', encoding="utf-8")
    rows, error = read_partial_checkpoint(path)
    assert rows == [{"question_id": "fixture"}]
    assert "line 2" in error and "not a result object" in error


def test_missing_pre_delivery_gate_refuses_before_protocol_or_any_writes(config, tmp_path, monkeypatch):
    import truthgate.validation_run as execution

    def forbidden_verify(*args):
        pytest.fail("Missing delivery must be rejected before protocol processing")

    monkeypatch.setattr(execution, "verify_protocol", forbidden_verify)
    with pytest.raises(ValueError, match="Deliver the pre-experiment PDF"):
        execute_validation(config, tmp_path / "manifest.json", tmp_path,
                           pre_report_delivered=False, runner=object())
    assert list(tmp_path.iterdir()) == []
