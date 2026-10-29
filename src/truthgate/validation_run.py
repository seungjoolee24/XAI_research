"""Frozen protocols and per-prompt checkpoints for larger baseline runs."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

from .dataset import build_prompt, load_questions
from .experiment import build_metadata, write_csv, write_json, write_records
from .metrics import attach_flip_metrics, summarize_results


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def planned_prompts(config: dict) -> list[tuple[Any, str, str]]:
    questions = load_questions(config["dataset"]["path"], config["dataset"]["include_swapped"])
    limit = config["experiment"]["limit"]
    if limit is not None:
        ids = list(dict.fromkeys(question.question_id for question in questions))[:limit]
        questions = [question for question in questions if question.question_id in ids]
    return [(question, condition, build_prompt(question, condition))
            for question in questions for condition in config["experiment"]["conditions"]]


def protocol_identity(config: dict, manifest_path: str | Path) -> dict:
    root = Path(config["_config_path"]).parent.parent
    sources = sorted([*root.glob("src/truthgate/*.py"), *root.glob("scripts/*.py")])
    return {
        "effective_config": config,
        "dataset_sha256": _sha(Path(config["dataset"]["path"])),
        "manifest_sha256": _sha(Path(manifest_path)),
        "sources": {path.relative_to(root).as_posix(): _sha(path) for path in sources},
        "prompts": [{"question_id": question.question_id, "option_order": question.option_order,
                     "condition": condition, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
                    for question, condition, prompt in planned_prompts(config)],
    }


def prepare_validation(config: dict, manifest_path: str | Path, directory: str | Path) -> Path:
    from .validation_reports import write_validation_pre_report

    if config["experiment"]["conditions"] != ["neutral", "false_opinion", "correct_opinion"]:
        raise ValueError("Expanded validation requires neutral, false_opinion, correct_opinion in that order.")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    pdf = write_validation_pre_report(config, manifest=manifest_path, directory=directory)
    identity = protocol_identity(config, manifest_path)
    write_json(directory / "execution_protocol.json", {
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "identity": identity, "identity_sha256": hashlib.sha256(_canonical(identity).encode()).hexdigest(),
        "pre_pdf_sha256": _sha(pdf),
    })
    return pdf


def verify_protocol(config: dict, manifest_path: str | Path, directory: str | Path) -> dict:
    directory = Path(directory)
    protocol = json.loads((directory / "execution_protocol.json").read_text(encoding="utf-8"))
    if _canonical(protocol["identity"]) != _canonical(protocol_identity(config, manifest_path)):
        raise ValueError("Dataset, manifest, effective config or source changed after the pre PDF. Prepare a new protocol.")
    if _sha(directory / "validation_pre_experiment.pdf") != protocol["pre_pdf_sha256"]:
        raise ValueError("The frozen pre-experiment PDF changed.")
    return protocol


def collect_validation(runner: Any, config: dict, directory: str | Path, *, resume: bool = False) -> list[dict]:
    """Append durable raw results; a resume accepts only exact planned records."""
    directory = Path(directory)
    plan = planned_prompts(config)
    planned = {(question.question_id, question.option_order, condition): (question, prompt)
               for question, condition, prompt in plan}
    checkpoint = directory / "checkpoint.jsonl"
    rows: list[dict] = []
    seen = set()
    if checkpoint.exists():
        if not resume:
            raise FileExistsError("Checkpoint exists; use --resume with the unchanged protocol.")
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            key = (row["question_id"], row["option_order"], row["condition"])
            if key not in planned or key in seen:
                raise ValueError("Checkpoint contains unknown or duplicate prompt keys.")
            question, prompt = planned[key]
            if (row["prompt"] != prompt or row["ground_truth"] != question.ground_truth
                    or row["option_a"] != question.options["A"] or row["option_b"] != question.options["B"]):
                raise ValueError("Checkpoint record does not match its frozen prompt.")
            seen.add(key)
            rows.append(row)
    # Check the complete tokenized protocol before the first research forward.
    for _, _, prompt in plan:
        runner.prepare_prompt(prompt)
    metadata_by_id = {row["question_id"]: row for row in json.loads(Path(config["dataset"]["path"]).read_text(encoding="utf-8"))}
    with checkpoint.open("a", encoding="utf-8", newline="\n") as handle:
        for question, condition, prompt in plan:
            key = (question.question_id, question.option_order, condition)
            if key in seen:
                continue
            result = runner.evaluate_prompt(prompt)
            sign = 1 if question.ground_truth == "A" else -1
            source = metadata_by_id[question.question_id]
            row = {
                "question_id": question.question_id, "option_order": question.option_order,
                "question": question.question, "option_a": question.options["A"], "option_b": question.options["B"],
                "ground_truth": question.ground_truth, "condition": condition,
                "subject": source.get("subject"), "domain": source.get("domain"),
                "curriculum_level": source.get("curriculum_level"),
                "user_opinion": None if condition == "neutral" else
                    (question.ground_truth if condition == "correct_opinion" else ("B" if question.ground_truth == "A" else "A")),
                **result, "truth_score_margin": sign * result["score_margin"],
                "truth_logit_margin": sign * result["logit_margin"] if result["logit_margin"] is not None else None,
            }
            handle.write(_canonical(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            rows.append(row)
            seen.add(key)
            if len(rows) == 1 or len(rows) % 10 == 0 or len(rows) == len(plan):
                progress = {"completed_rows": len(rows), "planned_rows": len(plan),
                            "last_question_id": question.question_id, "last_condition": condition,
                            "updated_at_utc": datetime.now(timezone.utc).isoformat()}
                write_json(directory / "progress.json", progress)
                print(f"PROGRESS {len(rows)}/{len(plan)} {question.question_id} {question.option_order} {condition}", flush=True)
    # Restore protocol order even when a resumed checkpoint was reordered.
    order = {key: index for index, key in enumerate(planned)}
    rows.sort(key=lambda row: order[(row["question_id"], row["option_order"], row["condition"])])
    return attach_flip_metrics(rows)


def runtime_identity(metadata: dict) -> dict:
    """Numerically relevant identity required for a mixed-attempt checkpoint."""
    keys = ("model_name", "model_revision", "tokenizer_revision", "chat_template_sha256",
            "dataset_sha256", "device", "dtype", "versions", "installed_distributions", "inference_settings")
    return {**{key: metadata.get(key) for key in keys},
            "runtime": {key: metadata.get("runtime", {}).get(key) for key in
                        ("platform", "cuda_runtime", "gpu_name", "cudnn_version", "deterministic_algorithms")}}


def read_partial_checkpoint(path: Path) -> tuple[list[dict], str | None]:
    """Read only the intact prefix for an honest failure report; never repair it."""
    rows = []
    if path.exists():
        for number, line in enumerate(path.read_bytes().splitlines(), 1):
            try:
                row = json.loads(line.decode("utf-8"))
                if not isinstance(row, dict):
                    raise ValueError("not a result object")
                rows.append(row)
            except (ValueError, TypeError) as error:
                return rows, f"Malformed checkpoint line {number}: {error}"
    return rows, None


def execute_validation(config: dict, manifest_path: str | Path, directory: str | Path, *,
                       pre_report_delivered: bool, resume: bool = False, runner: Any = None) -> dict:
    from .model import CausalLMRunner, set_seed
    from .validation import analyze_validation, write_validation_analysis
    from .validation_reports import write_validation_post_report

    directory = Path(directory)
    if not pre_report_delivered:
        raise ValueError("Deliver the pre-experiment PDF before starting inference.")
    verify_protocol(config, manifest_path, directory)
    if (directory / "completion.json").exists():
        raise FileExistsError("This run is complete; preserve it and prepare a new run directory.")
    checkpoint = directory / "checkpoint.jsonl"
    if checkpoint.exists() and not resume:
        raise FileExistsError("Checkpoint exists; use --resume with the unchanged protocol.")
    old_metadata_path = directory / "run_metadata.json"
    previous_metadata = json.loads(old_metadata_path.read_text(encoding="utf-8")) if old_metadata_path.exists() else None
    if checkpoint.exists() and checkpoint.stat().st_size and not previous_metadata:
        raise ValueError("Checkpoint has no runtime provenance; it cannot safely be resumed.")
    started = datetime.now(timezone.utc).isoformat()
    rows = None
    metadata = None
    attempt_root = directory / "attempts"
    attempt = attempt_root / f"{len(list(attempt_root.glob('*'))) + 1:03d}"
    attempt.mkdir(parents=True, exist_ok=False)
    try:
        set_seed(config["seed"])
        if runner is None:
            runner = CausalLMRunner.from_config(config["model"])
        metadata = build_metadata(config, runner, kind="expanded_validation")
        if checkpoint.exists() and checkpoint.stat().st_size and previous_metadata and runtime_identity(metadata) != runtime_identity(previous_metadata):
            write_json(attempt / "runtime_mismatch.json", {"previous": runtime_identity(previous_metadata), "attempted": runtime_identity(metadata)})
            metadata = previous_metadata
            raise ValueError("Runtime/model/tokenizer/dtype/dependencies changed; refuse to mix checkpoint results.")
        metadata["execution_started_at_utc"] = started
        metadata["attempt_directory"] = attempt.relative_to(directory).as_posix()
        write_json(attempt / "run_metadata.json", metadata)
        write_json(directory / "run_metadata.json", metadata)
        rows = collect_validation(runner, config, directory, resume=resume)
        verify_protocol(config, manifest_path, directory)
        write_records(directory, "results", rows)
        summary = summarize_results(rows)
        write_json(directory / "summary.json", summary)
        write_csv(directory / "summary.csv", [{"condition": key, **value} for key, value in summary["by_condition"].items()])
        selected_data = json.loads(Path(config["dataset"]["path"]).read_text(encoding="utf-8"))
        if config["experiment"]["limit"] is not None:
            selected_data = selected_data[:config["experiment"]["limit"]]
        analysis_options = {"require_swapped": config["dataset"]["include_swapped"]}
        analysis = analyze_validation(rows, selected_data, **analysis_options)
        write_validation_analysis(directory, rows, selected_data, **analysis_options)
        metadata["execution_finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        metadata["completed_rows"] = len(rows)
        metadata["status"] = "completed"
        write_json(directory / "run_metadata.json", metadata)
        write_json(attempt / "run_metadata.json", metadata)
        pdf = write_validation_post_report(config, manifest=manifest_path, directory=directory,
                                           analysis=analysis, metadata=metadata, rows=rows)
        write_json(directory / "completion.json", {"status": "completed", "n_rows": len(rows),
                   "post_report": pdf.name, "finished_at_utc": metadata["execution_finished_at_utc"]})
        shutil.copyfile(pdf, attempt / pdf.name)
        if (directory / "failure.json").exists():
            old_failure = json.loads((directory / "failure.json").read_text(encoding="utf-8"))
            old_failure["superseded_by_successful_resume"] = metadata["execution_finished_at_utc"]
            write_json(directory / "failure.json", old_failure)
        print(f"COMPLETED {len(rows)} rows; post PDF: {pdf}", flush=True)
        return analysis
    except (Exception, KeyboardInterrupt) as error:
        checkpoint_error = None
        if rows is None and checkpoint.exists():
            rows, checkpoint_error = read_partial_checkpoint(checkpoint)
        failure = {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                   "error_type": type(error).__name__, "error": str(error), "completed_rows": len(rows or []),
                   "started_at_utc": started, "ended_at_utc": datetime.now(timezone.utc).isoformat(),
                   "checkpoint_error": checkpoint_error, "attempt_directory": attempt.relative_to(directory).as_posix()}
        write_json(attempt / "failure.json", failure)
        write_json(directory / "failure.json", failure)
        pdf = write_validation_post_report(config, manifest=manifest_path, directory=directory,
                                     metadata=metadata, rows=rows, status=failure["status"],
                                     error=f"{type(error).__name__}: {error}")
        shutil.copyfile(pdf, attempt / pdf.name)
        raise
