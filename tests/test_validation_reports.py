"""Report provenance and actual-analysis denominator checks, with fixtures only."""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from pypdf import PdfReader
import pytest

from truthgate.experiment import load_config
from truthgate.validation import analyze_validation
from truthgate.validation_reports import validation_protocol_evidence, write_validation_post_report, write_validation_pre_report


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def protocol(tmp_path):
    dataset = [{"question_id": "fixture_1", "question": "Synthetic fixture question?", "options": {"A": "true", "B": "false"}, "ground_truth": "A", "subject": "fixture", "domain": "STEM", "source": {"dataset": "fixture"}}]
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")
    manifest = {"source_question_count": 1, "question_ids": ["fixture_1"], "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "subject_count": 1, "domain_counts": {"STEM": 1}}
    config = load_config(ROOT / "configs/expanded.yaml")
    config["dataset"]["path"] = str(path)
    return config, manifest, dataset


@pytest.mark.parametrize("field,value", [("dataset_sha256", "wrong"), ("source_question_count", 2), ("question_ids", ["wrong"]), ("domain_counts", {"STEM": 2})])
def test_pre_fingerprint_rejects_manifest_mismatch(protocol, field, value):
    config, manifest, _ = protocol
    manifest[field] = value
    with pytest.raises(ValueError, match="manifest"):
        validation_protocol_evidence(config, manifest=manifest)


def test_pre_protocol_exact_grid_and_hashes_without_inference(protocol, tmp_path):
    config, manifest, _ = protocol
    original = deepcopy(config)
    pdf = write_validation_pre_report(config, manifest=manifest, directory=tmp_path)
    evidence = json.loads((tmp_path / "validation_pre_evidence.json").read_text(encoding="utf-8"))
    assert evidence["plan"]["question_ids"] == ["fixture_1"]
    assert evidence["plan"]["evaluations"] == 6
    assert evidence["dataset_sha256"] == manifest["dataset_sha256"]
    assert "src/truthgate/validation_reports.py" in evidence["source_sha256"]
    assert config == original
    text = "\n".join(page.extract_text() for page in PdfReader(pdf).pages)
    assert "실행 전 계획" in text and "fixture_1" in text
    assert "candidate" in text and "cluster bootstrap" in text


def test_post_uses_analysis_real_key_denominators_and_partial_status(protocol, tmp_path):
    config, manifest, dataset = protocol
    rows = []
    for order in ("original", "swapped"):
        truth = "A" if order == "original" else "B"
        for condition in ("neutral", "false_opinion", "correct_opinion"):
            answer = ("B" if truth == "A" else "A") if condition == "false_opinion" else truth
            truth_margin = -2.0 if condition == "false_opinion" else 2.0
            margin = truth_margin * (1 if truth == "A" else -1)
            p_a = 1 / (1 + math.exp(-margin))
            rows.append({"question_id": "fixture_1", "option_order": order, "condition": condition,
                         "ground_truth": truth, "option_a": "true" if truth == "A" else "false", "option_b": "false" if truth == "A" else "true",
                         "predicted_answer": answer, "candidate_answer": answer, "generated_text": answer,
                         "score_margin": margin, "truth_score_margin": truth_margin,
                         "probability_a": p_a, "probability_b": 1-p_a,
                         "candidate_logprob_a": math.log(.8*p_a), "candidate_logprob_b": math.log(.8*(1-p_a)),
                         "candidates_are_single_tokens": True})
    analysis = analyze_validation(rows, dataset, bootstrap_repetitions=20)
    path = write_validation_post_report(config, manifest=manifest, directory=tmp_path, analysis=analysis, rows=rows)
    text = "\n".join(page.extract_text() for page in PdfReader(path).pages)
    assert "2/2" in text  # Actual harmful numerator / strict-neutral denominator.
    assert "-4.000" in text
    assert "0.800" in text and "[" in text
    assert "实际" not in text
    path = write_validation_post_report(config, manifest=manifest, directory=tmp_path / "partial", rows=rows[:1], status="interrupted", error="fixture interruption")
    partial = "\n".join(page.extract_text() for page in PdfReader(path).pages)
    assert "실패 / 중단 / 미완료" in partial and "fixture interruption" in partial
    assert "전체 조건별 결과" not in partial


def test_completed_post_requires_actual_nonempty_evidence(protocol, tmp_path):
    config, manifest, _ = protocol
    with pytest.raises(ValueError, match="actual nonempty"):
        write_validation_post_report(config, manifest=manifest, directory=tmp_path)
