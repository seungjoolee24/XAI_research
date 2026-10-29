"""Shared configuration, strict result serialization, and run provenance."""

from __future__ import annotations

import copy
import csv
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import io
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any
from uuid import uuid4

import yaml

from .dataset import CONDITIONS


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _conditions(value: Any, name: str, allowed: tuple[str, ...]) -> list[str]:
    if not isinstance(value, list) or not value or any(item not in allowed for item in value):
        raise ValueError(f"{name} must be a nonempty list selected from {allowed}")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} must contain unique conditions")
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    """Validate the baseline YAML schema and resolve paths beside its file.

    Device/dtype semantics are checked by the model loader. This helper checks
    their types without initializing CUDA or importing model dependencies.
    """
    config_path = Path(path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration must be a YAML mapping")
    seed = config.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an integer in [0, 2**32 - 1]")
    schema = {
        "model": ("name", "revision", "device", "dtype", "max_input_tokens", "max_new_tokens", "attention_implementation"),
        "dataset": ("path", "include_swapped"),
        "experiment": ("conditions", "limit"),
        "hidden_states": ("question_limit", "conditions"),
        "output": ("directory",),
    }
    for section, fields in schema.items():
        if not isinstance(config.get(section), dict):
            raise ValueError(f"{section} must be a mapping")
        missing = set(fields) - config[section].keys()
        if missing:
            raise ValueError(f"{section} is missing required fields: {sorted(missing)}")
    for key in ("name", "revision", "device", "dtype", "attention_implementation"):
        _text(config["model"][key], f"model.{key}")
    for key in ("max_input_tokens", "max_new_tokens"):
        _positive_int(config["model"][key], f"model.{key}")
    if not isinstance(config["dataset"]["include_swapped"], bool):
        raise ValueError("dataset.include_swapped must be a boolean")
    selected = _conditions(config["experiment"]["conditions"], "experiment.conditions", CONDITIONS)
    if "neutral" not in selected:
        raise ValueError("experiment.conditions must include neutral for paired metrics")
    limit = config["experiment"]["limit"]
    if limit is not None:
        _positive_int(limit, "experiment.limit")
    hidden_limit = _positive_int(config["hidden_states"]["question_limit"], "hidden_states.question_limit")
    if hidden_limit > 3:
        raise ValueError("hidden_states.question_limit must be at most 3 in Milestone 1")
    _conditions(config["hidden_states"]["conditions"], "hidden_states.conditions", ("false_opinion", "correct_opinion"))
    for section, name in (("dataset", "path"), ("output", "directory")):
        target = Path(_text(config[section][name], f"{section}.{name}")).expanduser()
        if not target.is_absolute():
            target = config_path.parent / target
        config[section][name] = str(target.resolve())
    config["_config_path"] = str(config_path)
    return config


def create_run_directory(config: dict[str, Any], kind: str, explicit: str | Path | None = None) -> Path:
    """Create a new directory exclusively so completed runs cannot be replaced."""
    if explicit is None:
        if not isinstance(kind, str) or not kind or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in kind):
            raise ValueError("run kind must contain only letters, digits, underscores, or hyphens")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        # Some platforms return the same wall-clock value for successive calls.
        # A random suffix keeps runs separate even within one clock tick.
        target = Path(config["output"]["directory"]) / f"{timestamp}_{kind}_{uuid4().hex[:12]}"
    else:
        target = Path(explicit).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=False)
    return target


def _json_text(value: Any, **options: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, **options)


def write_json(path: str | Path, obj: Any) -> None:
    """Write UTF-8 JSON, rejecting nonfinite numbers before touching the file."""
    content = _json_text(obj, indent=2) + "\n"
    Path(path).write_text(content, encoding="utf-8")


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    """Write a stable union header; nested cells use strict JSON and null is blank."""
    _json_text(rows)  # Reject NaN/Infinity and unsupported objects before writing.
    if fieldnames is None:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    output = io.StringIO(newline="")
    if fieldnames:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: _json_text(value) if isinstance(value, (dict, list, tuple)) else value
                for key, value in row.items()
            })
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        handle.write(output.getvalue())


def write_records(directory: str | Path, stem: str, rows: list[dict[str, Any]]) -> dict[str, Path]:
    """Write the same small result records to JSONL and a CSV with union columns."""
    if not stem or Path(stem).name != stem or stem in (".", ".."):
        raise ValueError("record stem must be a filename without a directory")
    directory = Path(directory)
    paths = {"jsonl": directory / f"{stem}.jsonl", "csv": directory / f"{stem}.csv"}
    jsonl = "".join(_json_text(row) + "\n" for row in rows)
    write_csv(paths["csv"], rows)
    paths["jsonl"].write_text(jsonl, encoding="utf-8")
    return paths


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _repository_root(config: dict[str, Any]) -> Path:
    config_path = Path(config["_config_path"])
    for directory in config_path.parents:
        if (directory / "pyproject.toml").is_file():
            return directory
    return config_path.parent.parent


def _git_metadata(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"repository_root": str(root), "commit": None, "dirty": None}
    try:
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "--verify", "HEAD"],
                                capture_output=True, text=True, timeout=5, check=False)
        status = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                                capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return result
    if commit.returncode == 0:
        result["commit"] = commit.stdout.strip()
    if status.returncode == 0:
        result["dirty"] = bool(status.stdout.strip())
    return result


def _versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {"python": platform.python_version()}
    for package in ("torch", "transformers", "accelerate", "numpy", "pandas", "PyYAML", "matplotlib"):
        try:
            result[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            result[package] = None
    return result


def _installed_distributions() -> dict[str, str]:
    """Capture transitive package versions in addition to the primary stack."""
    distributions = {}
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            distributions[name] = distribution.version
    return dict(sorted(distributions.items(), key=lambda item: item[0].casefold()))


def _runtime_metadata(device: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "platform": platform.platform(), "python_executable": sys.executable,
        "cuda_available": False, "cuda_runtime": None, "gpu_name": None,
        "cuda_device_count": 0, "cudnn_version": None,
    }
    try:
        import torch
    except ImportError:
        return result
    result["cuda_available"] = bool(torch.cuda.is_available())
    result["cuda_runtime"] = torch.version.cuda
    result["deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
    result["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
    result["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
    if result["cuda_available"]:
        selected = torch.device(device)
        index = selected.index if selected.type == "cuda" and selected.index is not None else 0
        result["gpu_name"] = torch.cuda.get_device_name(index)
        result["cuda_device_count"] = torch.cuda.device_count()
        result["cudnn_version"] = torch.backends.cudnn.version()
    return result


def build_metadata(config: dict[str, Any], runner: Any, *, kind: str) -> dict[str, Any]:
    """Record effective inputs and runtime identity without inventing outcomes."""
    model = runner.model
    model_config = model.config
    tokenizer = runner.tokenizer
    requested = config["model"]["revision"]
    parameter = next(model.parameters(), None)
    dtype = getattr(parameter, "dtype", getattr(model, "dtype", None))
    template = getattr(tokenizer, "chat_template", None)
    if template is not None and not isinstance(template, str):
        template = _json_text(template, sort_keys=True)
    init_kwargs = getattr(tokenizer, "init_kwargs", {}) or {}
    device = str(runner.device)
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "kind": kind,
        "effective_config": copy.deepcopy(config),
        "seed": config["seed"],
        "model_name": config["model"]["name"],
        "requested_revision": requested,
        "model_revision": getattr(model_config, "_commit_hash", None) or requested,
        "tokenizer_revision": init_kwargs.get("_commit_hash") or requested,
        "chat_template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest() if template is not None else None,
        "dataset_sha256": _file_sha256(Path(config["dataset"]["path"])),
        "device": device,
        "dtype": str(dtype) if dtype is not None else None,
        "model_num_hidden_layers": getattr(model_config, "num_hidden_layers", None),
        "model_hidden_size": getattr(model_config, "hidden_size", None),
        "versions": _versions(),
        "installed_distributions": _installed_distributions(),
        "runtime": _runtime_metadata(device),
        "git": _git_metadata(_repository_root(config)),
        "inference_settings": {
            "batch_size": 1, "do_sample": False, "num_beams": 1,
            "inference_mode": True, "model_eval": not getattr(model, "training", False),
            "max_input_tokens": runner.max_input_tokens,
            "max_new_tokens": runner.max_new_tokens,
            "trust_remote_code": False,
        },
        "candidate_probability_interpretation": "Probabilities normalize over only A/B candidate scores and are not calibrated factual confidence.",
    }
    _json_text(result)
    return result
