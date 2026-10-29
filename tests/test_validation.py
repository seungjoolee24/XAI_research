"""Offline deterministic fixtures for expanded A/B descriptive analysis."""

from copy import deepcopy
import json
import math

import pytest

from truthgate.validation import analyze_validation, parse_supplemental_answer, write_validation_analysis


@pytest.fixture
def experiment():
    dataset = [
        {"question_id": "q1", "question": "Fixture one?", "options": {"A": "true one", "B": "false one"}, "ground_truth": "A", "subject": "logic", "domain": "STEM", "source": {"dataset": "fixture"}},
        {"question_id": "q2", "question": "Fixture two?", "options": {"A": "false two", "B": "true two"}, "ground_truth": "B", "subject": "history", "domain": "humanities", "source": "fixture"},
    ]
    rows = []
    for item in dataset:
        for order in ("original", "swapped"):
            swapped = order == "swapped"
            truth = ("B" if item["ground_truth"] == "A" else "A") if swapped else item["ground_truth"]
            for condition in ("neutral", "false_opinion", "correct_opinion"):
                margin = 2.0 if truth == "A" else -2.0
                probability_a = 1 / (1 + math.exp(-margin))
                rows.append({
                    "question_id": item["question_id"], "option_order": order, "condition": condition,
                    "option_a": item["options"]["B" if swapped else "A"], "option_b": item["options"]["A" if swapped else "B"],
                    "ground_truth": truth, "predicted_answer": truth, "candidate_answer": truth,
                    "generated_text": truth, "generated_token_ids": [32 if truth == "A" else 33, 99],
                    "candidate_token_ids": {"A": [32], "B": [33]},
                    "score_margin": margin, "truth_score_margin": 2.0,
                    "candidate_logprob_a": math.log(.8 * probability_a), "candidate_logprob_b": math.log(.8 * (1 - probability_a)),
                    "probability_a": probability_a, "probability_b": 1 - probability_a,
                    "candidates_are_single_tokens": True,
                })
    return rows, dataset


def change(row, *, answer="keep", candidate="keep", truth_margin=None, text=None):
    if answer != "keep":
        row["predicted_answer"] = answer
        row["generated_text"] = answer if answer is not None else "I cannot tell."
        row["generated_token_ids"] = [32 if answer == "A" else 33 if answer == "B" else 98, 99]
    if candidate != "keep":
        if truth_margin is None:
            truth_margin = 0.0 if candidate is None else 2.0 if candidate == row["ground_truth"] else -2.0
    if truth_margin is not None:
        margin = truth_margin * (1 if row["ground_truth"] == "A" else -1)
        row["score_margin"] = margin
        row["truth_score_margin"] = truth_margin
        row["candidate_answer"] = "A" if margin > 0 else "B" if margin < 0 else None
        p_a = 1 / (1 + math.exp(-margin))
        row["probability_a"], row["probability_b"] = p_a, 1 - p_a
        row["candidate_logprob_a"], row["candidate_logprob_b"] = math.log(.8 * p_a), math.log(.8 * (1 - p_a))
    if text is not None:
        row["generated_text"] = text


def find(rows, qid, order, condition):
    return next(row for row in rows if (row["question_id"], row["option_order"], row["condition"]) == (qid, order, condition))


def test_denominators_invalids_and_raw_mass(experiment):
    rows, dataset = experiment
    for order in ("original", "swapped"):
        row = find(rows, "q1", order, "false_opinion")
        wrong = "B" if row["ground_truth"] == "A" else "A"
        change(row, answer=wrong, candidate=wrong)
    change(find(rows, "q2", "original", "false_opinion"), answer=None)
    result = analyze_validation(rows, dataset, bootstrap_repetitions=100)
    summary = result["by_condition"]["false_opinion"]
    assert summary["n"] == 4
    assert summary["source_question_count"] == 2
    assert summary["strict_correct_count"] == 1
    assert summary["strict_accuracy"] == .25
    assert summary["strict_invalid_count"] == 1
    assert summary["strict_harmful_flip_count"] == 2
    assert summary["strict_neutral_correct_count"] == 4
    assert summary["strict_harmful_flip_rate"] == .5
    assert summary["strict_harmful_flip_valid_denominator"] == 3
    assert summary["strict_harmful_flip_valid_rate"] == pytest.approx(2 / 3)
    assert summary["strict_invalid_given_neutral_correct_count"] == 1
    assert summary["candidate_accuracy"] == .5
    assert summary["ab_next_token_mass_mean"] == pytest.approx(.8)
    assert summary["mean_truth_margin_delta"] == -2
    assert summary["strict_beneficial_flip_rate"] is None
    assert len(result["by_subject"]) == 6
    assert len(result["by_domain"]) == 6


def test_beneficial_denominator_and_ties(experiment):
    rows, dataset = experiment
    row = find(rows, "q1", "original", "neutral")
    change(row, answer="B", candidate="B")
    change(find(rows, "q2", "original", "false_opinion"), candidate=None)
    result = analyze_validation(rows, dataset, bootstrap_repetitions=100)
    summary = result["by_condition"]["false_opinion"]
    assert summary["strict_beneficial_flip_count"] == 1
    assert summary["strict_neutral_incorrect_valid_count"] == 1
    assert summary["strict_beneficial_flip_rate"] == 1
    assert summary["candidate_tie_count"] == 1
    assert summary["candidate_harmful_flip_count"] == 0
    assert summary["candidate_pair_valid_count"] == 3


def test_candidate_only_changes_separate_unavailable_strict(experiment):
    rows, dataset = experiment
    change(find(rows, "q1", "original", "false_opinion"), candidate="B")
    change(find(rows, "q1", "swapped", "false_opinion"), answer=None, candidate="A")
    result = analyze_validation(rows, dataset, bootstrap_repetitions=100)
    summary = result["by_condition"]["false_opinion"]
    assert summary["candidate_only_change_count"] == 1
    assert summary["candidate_change_strict_unavailable_count"] == 1
    assert summary["strict_candidate_disagreement_count"] == 1


def test_swapped_semantic_consistency_and_invalids(experiment):
    rows, dataset = experiment
    # Same semantic answer has opposite A/B label after the verified swap.
    change(find(rows, "q1", "swapped", "false_opinion"), answer="A", candidate="A")
    change(find(rows, "q2", "original", "false_opinion"), answer=None)
    result = analyze_validation(rows, dataset, bootstrap_repetitions=100)
    order = {row["condition"]: row for row in result["order_consistency"]}
    assert order["neutral"]["strict_consistency_rate"] == 1
    assert order["neutral"]["candidate_consistent_count"] == 2
    assert order["false_opinion"]["strict_consistent_count"] == 0
    assert order["false_opinion"]["strict_eligible_count"] == 1
    assert order["false_opinion"]["strict_invalid_pair_count"] == 1
    assert order["false_opinion"]["candidate_consistent_count"] == 1
    assert order["false_opinion"]["candidate_same_label_different_content_count"] == 1


@pytest.mark.parametrize(("text", "expected"), [
    ("A", "A"), ("  B \n", "B"), ("A.", "A"), ("A. 17", "A"), ("A) 17", "A"),
    ("B. South", "B"), ("South", "B"), ("17", "A"),
    ("A. South", None), ("A or B", None), ("A because B is wrong", None),
    ("A. 17 and B. South", None), ("The answer is A", None), ("", None),
])
def test_supplemental_parse_is_conservative(text, expected):
    assert parse_supplemental_answer(text, "17", "South") == expected


def test_supplemental_never_replaces_strict_and_first_token_diagnostic(experiment):
    rows, dataset = experiment
    row = find(rows, "q1", "original", "neutral")
    change(row, answer=None, text="A. true one")
    row["generated_token_ids"] = [32, 42, 99]
    result = analyze_validation(rows, dataset, bootstrap_repetitions=100)
    summary = result["by_condition"]["neutral"]
    assert summary["strict_accuracy"] == .75
    assert summary["supplemental_accuracy"] == 1
    assert summary["first_label_vs_strict_mismatch_count"] == 1
    assert summary["first_label_vs_supplemental_mismatch_count"] == 0
    assert summary["first_token_vs_strict_mismatch_count"] == 1
    assert summary["first_token_vs_supplemental_mismatch_count"] == 0
    assert row["predicted_answer"] is None


def test_cluster_bootstrap_keeps_both_orders_together(experiment):
    rows, dataset = experiment
    # q1 has source-average delta -2, q2 +2. Independent-order resampling
    # would produce intermediate values unavailable to a two-source bootstrap.
    for qid, truth_margin in (("q1", 0), ("q2", 4)):
        for order in ("original", "swapped"):
            change(find(rows, qid, order, "false_opinion"), truth_margin=truth_margin)
    result = analyze_validation(rows, dataset, bootstrap_repetitions=2000)
    bootstrap = result["cluster_bootstrap"]["false_opinion"]["mean_truth_margin_delta"]
    assert bootstrap == {"estimate": 0, "ci_low": -2, "ci_high": 2, "valid_repetitions": 2000, "undefined_repetitions": 0}
    assert result["bootstrap_metadata"]["source_question_count"] == 2
    assert result["bootstrap_metadata"]["unit"] == "source question_id"
    sources = [row for row in result["source_pairs"] if row["condition"] == "false_opinion"]
    assert [row["mean_truth_margin_delta"] for row in sources] == [-2, 2]


def test_bootstrap_reproducible_and_row_order_independent(experiment):
    rows, dataset = experiment
    change(find(rows, "q1", "original", "false_opinion"), truth_margin=-4)
    first = analyze_validation(rows, dataset, bootstrap_seed=7, bootstrap_repetitions=300)
    second = analyze_validation(list(reversed(rows)), list(reversed(dataset)), bootstrap_seed=7, bootstrap_repetitions=300)
    assert first["cluster_bootstrap"] == second["cluster_bootstrap"]


def test_bootstrap_preserves_opposing_orders_within_each_source(experiment):
    rows, dataset = experiment
    for qid in ("q1", "q2"):
        change(find(rows, qid, "original", "false_opinion"), truth_margin=12)
        change(find(rows, qid, "swapped", "false_opinion"), truth_margin=-8)
    result = analyze_validation(rows, dataset, bootstrap_repetitions=200)
    interval = result["cluster_bootstrap"]["false_opinion"]["mean_truth_margin_delta"]
    assert interval["estimate"] == 0
    assert interval["ci_low"] == 0
    assert interval["ci_high"] == 0


def test_zero_denominator_resamples_are_reported(experiment):
    rows, dataset = experiment
    for order in ("original", "swapped"):
        change(find(rows, "q2", order, "neutral"), answer=None)
    result = analyze_validation(rows, dataset, bootstrap_repetitions=200)
    interval = result["cluster_bootstrap"]["false_opinion"]["strict_harmful_flip_rate"]
    assert 0 < interval["undefined_repetitions"] < 200
    assert interval["valid_repetitions"] + interval["undefined_repetitions"] == 200


def test_no_neutral_correct_gives_undefined_bootstrap_ratio(experiment):
    rows, dataset = experiment
    for row in rows:
        if row["condition"] == "neutral":
            wrong = "B" if row["ground_truth"] == "A" else "A"
            change(row, answer=wrong, candidate=wrong)
    result = analyze_validation(rows, dataset, bootstrap_repetitions=100)
    for prefix in ("strict", "candidate"):
        ci = result["cluster_bootstrap"]["false_opinion"][f"{prefix}_harmful_flip_rate"]
        assert ci["estimate"] is None
        assert ci["ci_low"] is None
        assert ci["ci_high"] is None
        assert ci["valid_repetitions"] == 0
        assert ci["undefined_repetitions"] == 100


def test_sequence_scores_do_not_claim_next_token_mass(experiment):
    rows, dataset = experiment
    for row in rows:
        row["candidates_are_single_tokens"] = False
    result = analyze_validation(rows, dataset, bootstrap_repetitions=100)
    summary = result["by_condition"]["neutral"]
    assert summary["ab_next_token_mass_mean"] is None
    assert summary["ab_next_token_mass_eligible_count"] == 0
    assert result["row_diagnostics"][0]["raw_probability_kind"] == "candidate_sequence_probability"


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "truth", "options", "unknown", "candidate", "nan", "bad_prob", "bad_logprob", "strict_parse", "wrong_probability", "wrong_logprob"])
def test_malformed_experiments_fail(experiment, mutation):
    rows, dataset = experiment
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(deepcopy(rows[0]))
    elif mutation == "truth":
        for row in rows:
            if row["question_id"] == "q1":
                row["ground_truth"] = "B"
    elif mutation == "options":
        rows[0]["option_a"] = "different"
    elif mutation == "unknown":
        rows[0]["question_id"] = "unknown"
    elif mutation == "candidate":
        rows[0]["candidate_answer"] = "B"
    elif mutation == "nan":
        rows[0]["candidate_logprob_a"] = math.nan
    elif mutation == "bad_prob":
        rows[0]["probability_a"] = .1
    elif mutation == "bad_logprob":
        rows[0]["candidate_logprob_a"] = .1
    elif mutation == "strict_parse":
        rows[0]["predicted_answer"] = None
    elif mutation == "wrong_probability":
        rows[0]["probability_a"], rows[0]["probability_b"] = .1, .9
    elif mutation == "wrong_logprob":
        rows[0]["candidate_logprob_a"] = -1
    with pytest.raises(ValueError):
        analyze_validation(rows, dataset, bootstrap_repetitions=100)


def test_missing_entire_question_and_duplicate_dataset_fail(experiment):
    rows, dataset = experiment
    with pytest.raises(ValueError, match="incomplete experiment grid"):
        analyze_validation([row for row in rows if row["question_id"] != "q1"], dataset)
    with pytest.raises(ValueError, match="duplicate dataset"):
        analyze_validation(rows, [*dataset, dataset[0]])


def test_original_order_only_must_be_explicit(experiment):
    rows, dataset = experiment
    original = [row for row in rows if row["option_order"] == "original"]
    with pytest.raises(ValueError, match="incomplete experiment grid"):
        analyze_validation(original, dataset)
    result = analyze_validation(original, dataset, require_swapped=False, bootstrap_repetitions=100)
    assert result["n_rows"] == 6
    assert result["order_consistency"] == []


@pytest.mark.parametrize("kwargs", [{"bootstrap_seed": True}, {"bootstrap_seed": -1}, {"bootstrap_repetitions": 0}, {"bootstrap_repetitions": True}, {"require_swapped": 1}])
def test_invalid_analysis_settings_fail(experiment, kwargs):
    rows, dataset = experiment
    with pytest.raises(ValueError):
        analyze_validation(rows, dataset, **kwargs)


def test_write_outputs_are_json_safe_and_summary_is_compact(experiment, tmp_path):
    rows, dataset = experiment
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")
    paths = write_validation_analysis(tmp_path / "analysis", rows, path, bootstrap_repetitions=100)
    assert set(paths) == {"summary", "by_condition", "by_subject", "by_domain", "source_pairs", "order_consistency", "row_diagnostics"}
    assert all(path.is_file() for path in paths.values())
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert summary["n_source_questions"] == 2
    assert "row_diagnostics" not in summary
    assert "source_pairs" not in summary
    assert summary["record_files"]["row_diagnostics"] == "validation_row_diagnostics.csv"
    assert "raw_probability_correct" in paths["row_diagnostics"].read_text(encoding="utf-8").splitlines()[0]
