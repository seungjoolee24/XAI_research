"""Offline config, output integrity, and metadata tests with a fake runner."""

import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from truthgate.experiment import build_metadata, create_run_directory, load_config, write_csv, write_json, write_records


def _config():
    return {
        "seed": 42,
        "model": {"name": "test/model", "revision": "requested", "device": "cpu", "dtype": "float32", "max_input_tokens": 64, "max_new_tokens": 8, "attention_implementation": "sdpa"},
        "dataset": {"path": "../data/questions.json", "include_swapped": False},
        "experiment": {"conditions": ["neutral", "false_opinion", "correct_opinion"], "limit": None},
        "hidden_states": {"question_limit": 3, "conditions": ["false_opinion", "correct_opinion"]},
        "output": {"directory": "../outputs"},
    }


def _save(tmp_path, config=None):
    directory = tmp_path / "project" / "configs"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "baseline.yaml"
    path.write_text(yaml.safe_dump(config if config is not None else _config()), encoding="utf-8")
    return path


def test_paths_resolve_beside_config_independent_of_working_directory(tmp_path, monkeypatch):
    path = _save(tmp_path)
    unrelated = tmp_path / "elsewhere"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    config = load_config(path)
    assert config["_config_path"] == str(path.resolve())
    assert config["dataset"]["path"] == str((path.parent.parent / "data" / "questions.json").resolve())
    assert config["output"]["directory"] == str((path.parent.parent / "outputs").resolve())


@pytest.mark.parametrize("section, key, value", [
    (None, "seed", True), (None, "seed", -1), (None, "seed", 2**32), (None, "seed", 1.5),
    ("experiment", "conditions", ["false_opinion"]),
    ("experiment", "conditions", ["neutral", "neutral"]),
    ("experiment", "conditions", ["neutral", "unknown"]),
    ("experiment", "conditions", "neutral"),
    ("experiment", "limit", True), ("experiment", "limit", 0),
    ("hidden_states", "question_limit", 4), ("hidden_states", "question_limit", 0),
    ("hidden_states", "question_limit", True),
    ("hidden_states", "conditions", ["neutral"]),
    ("hidden_states", "conditions", ["false_opinion", "false_opinion"]),
    ("hidden_states", "conditions", []),
    ("model", "max_new_tokens", 0), ("model", "max_input_tokens", True),
    ("model", "device", 0), ("model", "dtype", ""),
    ("dataset", "include_swapped", "false"), ("dataset", "path", ""),
    ("output", "directory", None),
])
def test_invalid_config_is_rejected(tmp_path, section, key, value):
    config = _config()
    (config if section is None else config[section])[key] = value
    with pytest.raises(ValueError):
        load_config(_save(tmp_path, config))


@pytest.mark.parametrize("seed", [0, 2**32 - 1])
def test_seed_boundary_values_are_allowed(tmp_path, seed):
    config = _config()
    config["seed"] = seed
    assert load_config(_save(tmp_path, config))["seed"] == seed


def test_missing_sections_and_fields_are_rejected(tmp_path):
    config = _config()
    del config["model"]["revision"]
    with pytest.raises(ValueError, match="missing"):
        load_config(_save(tmp_path, config))
    del config["model"]
    with pytest.raises(ValueError, match="mapping"):
        load_config(_save(tmp_path, config))


def test_run_directories_are_unique_and_explicit_runs_cannot_be_overwritten(tmp_path, monkeypatch):
    class FrozenClock:
        @staticmethod
        def now(tz):
            return datetime(2026, 10, 29, tzinfo=timezone.utc)

    monkeypatch.setattr("truthgate.experiment.datetime", FrozenClock)
    config = load_config(_save(tmp_path))
    first = create_run_directory(config, "baseline")
    second = create_run_directory(config, "baseline")
    assert first.is_dir() and second.is_dir() and first != second
    explicit = tmp_path / "explicit"
    assert create_run_directory(config, "baseline", explicit) == explicit.resolve()
    sentinel = explicit / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        create_run_directory(config, "baseline", explicit)
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_jsonl_and_csv_preserve_nested_results_and_invalid_null(tmp_path):
    rows = [{"answer": None, "tokens": [1, 2], "scores": {"A": 0.2}, "text": "한글"}, {"answer": "A", "extra": True}]
    paths = write_records(tmp_path, "results", rows)
    assert [json.loads(line) for line in paths["jsonl"].read_text(encoding="utf-8").splitlines()] == rows
    with paths["csv"].open(encoding="utf-8", newline="") as handle:
        parsed = list(csv.DictReader(handle))
    assert parsed[0]["answer"] == ""
    assert json.loads(parsed[0]["tokens"]) == [1, 2]
    assert json.loads(parsed[0]["scores"]) == {"A": 0.2}
    assert parsed[0]["text"] == "한글"
    assert parsed[1]["extra"] == "True"


def test_empty_csv_can_retain_header_and_empty_records_are_valid(tmp_path):
    path = tmp_path / "flips.csv"
    write_csv(path, [], fieldnames=["question_id", "condition"])
    assert path.read_text(encoding="utf-8").splitlines() == ["question_id,condition"]
    paths = write_records(tmp_path, "empty", [])
    assert paths["jsonl"].read_text(encoding="utf-8") == ""
    assert paths["csv"].read_text(encoding="utf-8") == ""


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_results_do_not_truncate_existing_files(tmp_path, value):
    target = tmp_path / "existing.json"
    target.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError):
        write_json(target, {"value": value})
    assert target.read_text(encoding="utf-8") == "keep"
    with pytest.raises(ValueError):
        write_csv(tmp_path / "bad.csv", [{"value": value}])
    with pytest.raises(ValueError):
        write_records(tmp_path, "bad", [{"value": value}])
    assert not (tmp_path / "bad.csv").exists()
    assert not (tmp_path / "bad.jsonl").exists()


def test_metadata_records_actual_revisions_digest_and_effective_settings(tmp_path, monkeypatch):
    import truthgate.experiment as experiment

    config = load_config(_save(tmp_path))
    dataset = Path(config["dataset"]["path"])
    dataset.parent.mkdir()
    dataset.write_text("[]\n", encoding="utf-8")
    model_config = SimpleNamespace(_commit_hash="actual-model", num_hidden_layers=2, hidden_size=16)
    model = SimpleNamespace(config=model_config, parameters=lambda: iter([SimpleNamespace(dtype="torch.float32")]), training=False)
    tokenizer = SimpleNamespace(chat_template="TEST TEMPLATE", init_kwargs={"_commit_hash": "actual-tokenizer"})
    runner = SimpleNamespace(model=model, tokenizer=tokenizer, device="cpu", max_input_tokens=64, max_new_tokens=8)
    monkeypatch.setattr(experiment, "_runtime_metadata", lambda device: {"cuda_available": False, "gpu_name": None})
    monkeypatch.setattr(experiment, "_git_metadata", lambda root: {"commit": "git-sha", "dirty": True})
    result = build_metadata(config, runner, kind="baseline")
    assert result["requested_revision"] == "requested"
    assert result["model_revision"] == "actual-model"
    assert result["tokenizer_revision"] == "actual-tokenizer"
    assert result["dataset_sha256"] == hashlib.sha256(dataset.read_bytes()).hexdigest()
    assert result["chat_template_sha256"] == hashlib.sha256(b"TEST TEMPLATE").hexdigest()
    assert result["dtype"] == "torch.float32" and result["device"] == "cpu"
    assert result["model_num_hidden_layers"] == 2 and result["model_hidden_size"] == 16
    assert result["inference_settings"]["do_sample"] is False
    assert result["inference_settings"]["max_new_tokens"] == 8
    assert isinstance(result["installed_distributions"], dict)
    assert "not calibrated" in result["candidate_probability_interpretation"]
    json.dumps(result, allow_nan=False)
    config["seed"] = 99
    assert result["effective_config"]["seed"] == 42


def test_revision_fallback_and_missing_dependencies_remain_json_safe(tmp_path, monkeypatch):
    import truthgate.experiment as experiment

    config = load_config(_save(tmp_path))
    dataset = Path(config["dataset"]["path"])
    dataset.parent.mkdir()
    dataset.write_text("[]", encoding="utf-8")
    model = SimpleNamespace(config=SimpleNamespace(), parameters=lambda: iter([]), training=False)
    tokenizer = SimpleNamespace(chat_template=None, init_kwargs={})
    runner = SimpleNamespace(model=model, tokenizer=tokenizer, device="cpu", max_input_tokens=64, max_new_tokens=8)
    monkeypatch.setattr(experiment, "_runtime_metadata", lambda device: {"cuda_available": False})
    monkeypatch.setattr(experiment, "_git_metadata", lambda root: {"commit": None, "dirty": None})
    def missing_package(name):
        raise experiment.metadata.PackageNotFoundError(name)
    monkeypatch.setattr(experiment.metadata, "version", missing_package)
    result = build_metadata(copy.deepcopy(config), runner, kind="test")
    assert result["model_revision"] == "requested"
    assert result["tokenizer_revision"] == "requested"
    assert result["chat_template_sha256"] is None
    assert result["versions"]["torch"] is None
    json.dumps(result, allow_nan=False)
