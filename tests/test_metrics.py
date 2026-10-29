"""Pairing and invalid-answer regression tests without a model dependency."""

import json

import pytest

from truthgate.metrics import attach_flip_metrics, summarize_results


def _row(question_id="q1", condition="neutral", answer="A", candidate="A", truth="A", order="original"):
    return {
        "question_id": question_id,
        "option_order": order,
        "condition": condition,
        "ground_truth": truth,
        "predicted_answer": answer,
        "candidate_answer": candidate,
    }


def test_pairs_are_order_independent_and_do_not_mutate_input():
    rows = [_row(condition="false_opinion", answer="B"), _row(), _row(condition="correct_opinion")]
    enriched = attach_flip_metrics(rows)
    assert "is_correct" not in rows[0]
    assert enriched[0]["neutral_answer"] == "A"
    assert enriched[0]["answer_flip_from_neutral"] is True
    assert enriched[0]["correct_to_incorrect"] is True
    assert enriched[0]["incorrect_to_correct"] is False
    assert enriched[0]["candidate_flip_from_neutral"] is False
    assert enriched[1]["answer_flip_from_neutral"] is False
    assert enriched[2]["is_correct"] is True


def test_incorrect_to_correct_is_distinct_from_wrong_direction():
    enriched = attach_flip_metrics([
        _row(answer="B", candidate="B"),
        _row(condition="false_opinion", answer="A", candidate="A"),
    ])
    assert enriched[1]["incorrect_to_correct"] is True
    assert enriched[1]["correct_to_incorrect"] is False
    assert enriched[1]["candidate_incorrect_to_correct"] is True


def test_order_variants_have_separate_neutral_baselines():
    enriched = attach_flip_metrics([
        _row(),
        _row(condition="false_opinion", answer="B"),
        _row(order="swapped", answer="B", candidate="B", truth="B"),
        _row(condition="false_opinion", order="swapped", answer="B", candidate="B", truth="B"),
    ])
    assert enriched[1]["answer_flip_from_neutral"] is True
    assert enriched[3]["answer_flip_from_neutral"] is False
    assert enriched[3]["neutral_answer"] == "B"


@pytest.mark.parametrize("neutral_answer, biased_answer", [(None, "A"), ("A", None), (None, None)])
def test_invalid_greedy_answers_are_not_flips(neutral_answer, biased_answer):
    enriched = attach_flip_metrics([
        _row(answer=neutral_answer),
        _row(condition="false_opinion", answer=biased_answer),
    ])
    assert enriched[1]["answer_flip_from_neutral"] is None
    assert enriched[1]["correct_to_incorrect"] is None
    assert enriched[1]["incorrect_to_correct"] is None
    if biased_answer is None:
        assert enriched[1]["is_correct"] is False


def test_candidate_ties_are_invalid_and_independent_of_greedy():
    enriched = attach_flip_metrics([_row(candidate=None), _row(condition="false_opinion", answer="B")])
    assert enriched[0]["candidate_is_correct"] is False
    assert enriched[0]["candidate_flip_from_neutral"] is None
    assert enriched[1]["candidate_flip_from_neutral"] is None
    assert enriched[1]["candidate_correct_to_incorrect"] is None
    assert enriched[1]["correct_to_incorrect"] is True


@pytest.mark.parametrize("rows, message", [
    ([_row(condition="false_opinion")], "missing neutral"),
    ([_row(), _row()], "duplicate"),
    ([_row(), _row(condition="false_opinion"), _row(condition="false_opinion")], "duplicate"),
    ([_row(), _row(condition="false_opinion", truth="B")], "inconsistent"),
    ([_row(), _row(condition="false_opinion", order="swapped", truth="B")], "missing neutral"),
])
def test_ambiguous_pairing_is_rejected(rows, message):
    with pytest.raises(ValueError, match=message):
        attach_flip_metrics(rows)
    with pytest.raises(ValueError, match=message):
        summarize_results(rows)


@pytest.mark.parametrize("field, value", [
    ("question_id", " "), ("condition", "other"), ("option_order", "other"),
    ("ground_truth", None), ("predicted_answer", "C"), ("candidate_answer", "tie"),
])
def test_invalid_input_contract_fails(field, value):
    row = _row()
    row[field] = value
    with pytest.raises(ValueError):
        attach_flip_metrics([row])


def test_missing_candidate_is_rejected():
    row = _row()
    del row["candidate_answer"]
    with pytest.raises(ValueError, match="must contain"):
        attach_flip_metrics([row])


def test_summary_exposes_accuracy_and_both_directional_denominators():
    rows = [
        _row("q1"),
        _row("q1", "false_opinion", "B", "B"),
        _row("q2"),
        _row("q2", "false_opinion", None, None),
        _row("q3", answer="B", candidate="B"),
        _row("q3", "false_opinion", "A", "A"),
        _row("q4", answer=None, candidate=None),
        _row("q4", "false_opinion", "A", "A"),
    ]
    summary = summarize_results(rows)
    false = summary["by_condition"]["false_opinion"]
    assert summary["n_rows"] == 8
    assert summary["n_question_variants"] == 4
    assert false["n"] == 4
    assert false["accuracy"] == 0.5
    assert false["invalid_count"] == 1
    assert false["candidate_accuracy"] == 0.5
    assert false["candidate_invalid_count"] == 1
    assert false["answer_flip_count"] == 2
    assert false["answer_flip_eligible_count"] == 2
    assert false["answer_flip_rate"] == 1.0
    assert false["correct_to_incorrect_count"] == 1
    assert false["incorrect_to_correct_count"] == 1
    assert false["neutral_correct_count"] == 2
    assert false["correct_to_incorrect_rate"] == 0.5
    assert false["correct_to_incorrect_valid_eligible_count"] == 1
    assert false["correct_to_incorrect_valid_rate"] == 1.0
    assert false["invalid_given_neutral_correct_count"] == 1
    assert false["candidate_correct_to_incorrect_rate"] == 0.5
    assert "causal" in summary["metric_definitions"]["interpretation"]
    json.dumps(summary, allow_nan=False)


def test_empty_inputs_are_json_safe_and_use_null_for_undefined_rates():
    assert attach_flip_metrics([]) == []
    summary = summarize_results([])
    assert summary["n_rows"] == 0
    for metrics in summary["by_condition"].values():
        assert metrics["n"] == 0
        assert metrics["accuracy"] is None
        assert metrics["answer_flip_rate"] is None
        assert metrics["correct_to_incorrect_rate"] is None
    json.dumps(summary, allow_nan=False)
