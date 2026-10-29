"""Compare neutral, false-opinion, and correct-opinion prompts."""

import argparse
from pathlib import Path

import pandas as pd

from truthgate.dataset import build_prompt, load_questions
from truthgate.experiment import build_metadata, create_run_directory, load_config, write_csv, write_json, write_records
from truthgate.metrics import attach_flip_metrics, summarize_results
from truthgate.model import CausalLMRunner, set_seed
from truthgate.reports import write_pre_report, write_post_report


def run_baseline(runner: CausalLMRunner, config: dict) -> list[dict]:
    # Limit source questions before adding swapped variants, so --limit 1
    # --include-swapped evaluates both orders of that same question.
    questions = load_questions(config["dataset"]["path"], include_swapped=config["dataset"]["include_swapped"])
    limit = config["experiment"]["limit"]
    if limit is not None:
        selected_ids = list(dict.fromkeys(q.question_id for q in questions))[:limit]
        questions = [q for q in questions if q.question_id in selected_ids]
    rows = []
    for index, question in enumerate(questions, start=1):
        for condition in config["experiment"]["conditions"]:
            print(f"[{index}/{len(questions)}] {question.question_id} {question.option_order} {condition}", flush=True)
            result = runner.evaluate_prompt(build_prompt(question, condition))
            sign = 1 if question.ground_truth == "A" else -1
            rows.append({
                "question_id": question.question_id, "option_order": question.option_order,
                "question": question.question, "option_a": question.options["A"], "option_b": question.options["B"],
                "ground_truth": question.ground_truth, "condition": condition,
                "user_opinion": None if condition == "neutral" else (question.ground_truth if condition == "correct_opinion" else ("B" if question.ground_truth == "A" else "A")),
                **result,
                "truth_score_margin": sign * result["score_margin"],
                "truth_logit_margin": sign * result["logit_margin"] if result["logit_margin"] is not None else None,
            })
    return attach_flip_metrics(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/baseline.yaml")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--include-swapped", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        config["experiment"]["limit"] = args.limit
    if args.include_swapped:
        config["dataset"]["include_swapped"] = True
    if args.device:
        config["model"]["device"] = args.device
    directory = create_run_directory(config, "baseline", args.output_dir)
    pre_report = write_pre_report(config, kind="baseline", directory=directory)
    print(f"Pre-experiment PDF (before model load): {pre_report}", flush=True)
    rows = None
    metadata = None
    try:
        set_seed(config["seed"])
        runner = CausalLMRunner.from_config(config["model"])
        rows = run_baseline(runner, config)
        write_records(directory, "results", rows)
        summary = summarize_results(rows)
        write_json(directory / "summary.json", summary)
        summary_rows = [{"condition": condition, **metrics} for condition, metrics in summary["by_condition"].items()]
        write_csv(directory / "summary.csv", summary_rows)
        flips = [row for row in rows if row["condition"] != "neutral" and row["answer_flip_from_neutral"] is True]
        write_csv(directory / "flip_cases.csv", flips, fieldnames=list(rows[0]))
        metadata = build_metadata(config, runner, kind="baseline")
        write_json(directory / "run_metadata.json", metadata)
        post_report = write_post_report(config, kind="baseline", directory=directory, rows=rows, metadata=metadata)
    except (Exception, KeyboardInterrupt) as error:
        failure = {"status": "failed", "error_type": type(error).__name__, "error": str(error), "effective_config": config}
        write_json(directory / "failure.json", failure)
        post_report = write_post_report(config, kind="baseline", directory=directory, rows=rows, metadata=metadata, error=f"{type(error).__name__}: {error}")
        print(f"Failure report: {post_report}", flush=True)
        raise
    print(pd.DataFrame(summary_rows)[["condition", "n", "accuracy", "invalid_count", "candidate_accuracy", "answer_flip_count", "correct_to_incorrect_count"]].to_string(index=False))
    if flips:
        print("\nObserved greedy answer changes:")
        print(pd.DataFrame(flips)[["question_id", "option_order", "condition", "ground_truth", "neutral_answer", "predicted_answer", "correct_to_incorrect"]].to_string(index=False))
    else:
        print("\nNo valid greedy answer flips were observed in this run.")
    print(f"Post-experiment PDF: {post_report}")
    print(f"Saved baseline: {directory}")


if __name__ == "__main__":
    main()
