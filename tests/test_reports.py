"""Offline report checks; synthetic records are never research model results."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from pypdf import PdfReader
import pytest

from truthgate.experiment import load_config
from truthgate.reports import wilson_interval, write_post_report, write_pre_report


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    return load_config(ROOT / "configs/baseline.yaml")


def pdf_text(path):
    reader = PdfReader(path)
    assert all(float(p.mediabox.width) > 590 for p in reader.pages)
    return "\n".join(page.extract_text() for page in reader.pages)


def test_pre_is_korean_protocol_and_has_no_invented_outcomes(config, tmp_path):
    before = copy.deepcopy(config)
    path = write_pre_report(config, kind="baseline", directory=tmp_path)
    text = pdf_text(path)
    assert "실험 사전 설명서" in text
    assert "실행 전 계획" in text
    assert "20" in text and "60" in text
    assert "question_id, option_order" in text
    assert hashlib.sha256(Path(config["dataset"]["path"]).read_bytes()).hexdigest() in text.replace("\n", "")
    assert config == before
    assert len(PdfReader(path).pages) >= 3


def test_limit_is_source_count_before_swaps(config, tmp_path):
    config["experiment"]["limit"] = 1
    config["dataset"]["include_swapped"] = True
    text = pdf_text(write_pre_report(config, kind="baseline", directory=tmp_path))
    assert "6" in text and "True" in text


def test_wilson_empty_extreme_and_validation():
    assert wilson_interval(0, 0) is None
    low, high = wilson_interval(10, 10)
    assert low == pytest.approx(0.7224672)
    assert high == pytest.approx(1)
    assert wilson_interval(0, 10)[0] == pytest.approx(0)
    with pytest.raises(ValueError):
        wilson_interval(2, 1)


def test_post_separates_harmful_beneficial_invalid_candidate_and_margins(config, tmp_path):
    rows = []
    # q1: neutral correct -> false wrong, q2: neutral wrong -> false
    # correct, q3: false invalid. Candidate q3 changes independently.
    for qid, truth, neutral, biased, candidate in (
        ("q1", "A", "A", "B", "B"), ("q2", "A", "B", "A", "A"),
        ("q3", "A", "A", None, "B"),
    ):
        for condition, answer, cand, margin in (
            ("neutral", neutral, neutral, 1.0), ("false_opinion", biased, candidate, -1.0),
        ):
            rows.append({"question_id": qid, "option_order": "original", "condition": condition,
                         "ground_truth": truth, "predicted_answer": answer, "candidate_answer": cand,
                         "truth_score_margin": margin, "truth_logit_margin": margin / 2})
    text = pdf_text(write_post_report(config, kind="baseline", directory=tmp_path, rows=rows))
    assert "1/2" in text  # harmful / all neutral correct (invalid remains in denominator)
    assert "1/1" in text  # harmful / valid neutral-correct pairs
    assert "harmful" in text and "beneficial" in text
    assert "truth_score_margin" in text and "truth_logit_margin" in text
    assert "Qwen3-4B GPU 결과 (" not in text
    assert "provenance 미제공" in text


def test_failure_does_not_claim_success_or_accuracy(config, tmp_path):
    text = pdf_text(write_post_report(config, kind="baseline", directory=tmp_path, error="CUDA out of memory"))
    assert "실패 / 미완료" in text
    assert "CUDA out of memory" in text
    assert "생성 답변 정확도" not in text


def test_success_requires_actual_records(config, tmp_path):
    with pytest.raises(ValueError, match="actual nonempty"):
        write_post_report(config, kind="baseline", directory=tmp_path, rows=[])
    with pytest.raises(ValueError, match="actual nonempty"):
        write_post_report(config, kind="smoke", directory=tmp_path)


@pytest.mark.parametrize("kind", ["smoke", "hidden"])
def test_other_pre_reports_render_without_model_loading(config, tmp_path, kind):
    text = pdf_text(write_pre_report(config, kind=kind, directory=tmp_path))
    assert "실행 전 계획" in text
    assert "의사코드" in text


def test_hidden_post_retains_undefined_cosines(config, tmp_path):
    rows = [{"question_id": "q1", "option_order": "original", "condition": "false_opinion",
             "state_index": 0, "cosine_similarity": None, "neutral_last_token_norm": 0.0,
             "biased_last_token_norm": 1.0}]
    text = pdf_text(write_post_report(config, kind="hidden", directory=tmp_path, rows=rows))
    assert "N/A" in text and "zero vector" in text


def test_smoke_post_does_not_guess_ground_truth(config, tmp_path):
    text = pdf_text(write_post_report(config, kind="smoke", directory=tmp_path,
                                    result={"predicted_answer": "A", "generated_text": "A"}))
    assert "정답 라벨을 검증하지 않았으므로" in text


def test_unknown_kind_fails_before_pdf(config, tmp_path):
    with pytest.raises(ValueError, match="kind must"):
        write_pre_report(config, kind="training", directory=tmp_path)


def test_failed_run_cli_regeneration_uses_saved_failure_without_result_files(config, tmp_path):
    run_dir = tmp_path / "failed_run"
    run_dir.mkdir()
    config["experiment"]["limit"] = 1
    (run_dir / "failure.json").write_text(json.dumps({"status": "failed", "error_type": "RuntimeError",
        "error": "CUDA out of memory", "effective_config": config}), encoding="utf-8")
    output = tmp_path / "regenerated"
    completed = subprocess.run([sys.executable, str(ROOT / "scripts/generate_experiment_report.py"),
        "--config", str(ROOT / "configs/baseline.yaml"), "--kind", "baseline", "--phase", "post",
        "--run-dir", str(run_dir), "--output-dir", str(output)], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    text = pdf_text(output / "post_experiment.pdf")
    assert "CUDA out of memory" in text and "실패 / 미완료" in text
