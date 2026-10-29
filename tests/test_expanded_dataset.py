"""Offline checks of source selection and binary projection, not model science."""

from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from truthgate.dataset import load_questions


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("expanded_builder", ROOT / "scripts" / "build_expanded_dataset.py")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def fixture_rows(count=30):
    return [{"question": f"Independent fixture problem {index}?", "choices": [f"choice {index}-{j}" for j in range(4)], "answer": index % 4} for index in range(count)]


def transform(rows, **kwargs):
    return builder.transform_subject(rows, "fixture", "STEM", "unspecified", source_file="fixture/test.parquet", source_file_sha256="0" * 64, **kwargs)


def test_projection_is_deterministic_balanced_and_preserves_source_answer():
    first, stats = transform(fixture_rows())
    assert transform(fixture_rows()) == (first, stats)
    assert len(first) == 20
    assert Counter(row["ground_truth"] for row in first) == {"A": 10, "B": 10}
    for row in first:
        source = row["source"]
        assert row["options"][row["ground_truth"]] == source["choices"][source["answer_index"]]
        assert row["transformation"]["distractor_index"] != source["answer_index"]
        assert len(row["options"]) == 2
        assert source["revision"] == builder.DATASET_REVISION
    assert transform(fixture_rows(), seed=43)[0] != first


@pytest.mark.parametrize("choice", ["All of the above", "None of these", "Both A and B", "Neither of the above", "Options A and C", "(A) and (D)", "All other answers"])
def test_reference_dependent_options_are_rejected_before_sampling(choice):
    row = fixture_rows(1)[0]
    row["choices"][2] = choice
    assert builder.exclusion_reason(row) == "choice_reference_or_collective_option"


@pytest.mark.parametrize("update,reason", [
    ({"answer": 4}, "invalid_answer_index"),
    ({"answer": True}, "invalid_answer_index"),
    ({"choices": ["one", " ONE ", "two", "three"]}, "duplicate_choice_text"),
    ({"choices": ["one", "two"]}, "not_four_choices"),
    ({"question": "Use the following diagram to calculate the force."}, "potential_missing_figure"),
])
def test_bad_or_unreconstructable_source_items_are_rejected(update, reason):
    row = fixture_rows(1)[0]
    row.update(update)
    assert builder.exclusion_reason(row) == reason


def test_filter_precedes_sampling_and_duplicate_wording_is_excluded():
    rows = fixture_rows()
    rows.append(dict(rows[0]))
    rows.append({"question": "Bad reference?", "choices": ["first", "second", "third", "All of the above"], "answer": 0})
    selected, summary = transform(rows)
    assert summary["eligible_rows"] == 30
    assert summary["excluded_reasons"] == {"duplicate_question_text": 1, "choice_reference_or_collective_option": 1}
    assert len({row["question"] for row in selected}) == 20
    with pytest.raises(ValueError, match="eligible"):
        transform(fixture_rows(19))
    with pytest.raises(ValueError, match="positive even"):
        transform(fixture_rows(), count=3)


def test_committed_dataset_has_exact_breadth_source_evidence_and_manifest_digest():
    path = ROOT / "data" / "expanded_questions.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "data" / "expanded_dataset_manifest.json").read_text(encoding="utf-8"))
    assert len(rows) == 400
    assert len(load_questions(path, include_swapped=True)) == 800
    assert Counter(row["subject"] for row in rows) == {subject: 20 for subject, _, _ in builder.SUBJECTS}
    for subject, _, _ in builder.SUBJECTS:
        assert Counter(row["ground_truth"] for row in rows if row["subject"] == subject) == {"A": 10, "B": 10}
    assert manifest["dataset_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert manifest["question_ids"] == [row["question_id"] for row in rows]
    assert manifest["source_revision"] == builder.DATASET_REVISION
    assert manifest["license"] == "MIT"
    assert manifest["planned_prompt_rows"] == 2400
    for row in rows:
        source = row["source"]
        source_record = {"question": source["question"], "choices": source["choices"], "answer": source["answer_index"], "subject": source["subject"]}
        assert builder.exclusion_reason(source_record) is None
        assert source["source_record_sha256"] == builder.sha256_bytes(builder.canonical_record_bytes(source_record))
        assert row["options"][row["ground_truth"]] == source["choices"][source["answer_index"]]
        assert len(source["source_file_sha256"]) == 64
