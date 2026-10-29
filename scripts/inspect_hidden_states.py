"""Inspect last-prompt-token hidden states for up to three source questions."""

import argparse
from pathlib import Path

from truthgate.activations import compare_hidden_states, extract_last_token_states, plot_hidden_comparison
from truthgate.dataset import build_prompt, load_questions
from truthgate.experiment import build_metadata, create_run_directory, load_config, write_json, write_records
from truthgate.model import CausalLMRunner, set_seed
from truthgate.reports import write_pre_report, write_post_report


def inspect_hidden_states(runner: CausalLMRunner, config: dict) -> list[dict]:
    questions = load_questions(config["dataset"]["path"], include_swapped=config["dataset"]["include_swapped"])
    selected_ids = list(dict.fromkeys(q.question_id for q in questions))[:config["hidden_states"]["question_limit"]]
    questions = [q for q in questions if q.question_id in selected_ids]
    rows = []
    for question in questions:
        print(f"Hidden states: {question.question_id} {question.option_order}", flush=True)
        neutral = extract_last_token_states(runner, build_prompt(question, "neutral"))
        for condition in config["hidden_states"]["conditions"]:
            biased = extract_last_token_states(runner, build_prompt(question, condition))
            comparisons = compare_hidden_states(neutral, biased, question_id=question.question_id,
                                                option_order=question.option_order, condition=condition)
            for row in comparisons:
                row["ground_truth"] = question.ground_truth
            rows.extend(comparisons)
            del biased
        del neutral
    return rows


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
        if not 1 <= args.limit <= 3:
            parser.error("Hidden-state --limit must be between 1 and 3")
        config["hidden_states"]["question_limit"] = args.limit
    if args.include_swapped:
        config["dataset"]["include_swapped"] = True
    if args.device:
        config["model"]["device"] = args.device
    directory = create_run_directory(config, "hidden", args.output_dir)
    pre_report = write_pre_report(config, kind="hidden", directory=directory)
    print(f"Pre-experiment PDF (before model load): {pre_report}", flush=True)
    rows = None
    metadata = None
    try:
        set_seed(config["seed"])
        runner = CausalLMRunner.from_config(config["model"])
        rows = inspect_hidden_states(runner, config)
        write_records(directory, "hidden_states", rows)
        plot_hidden_comparison(rows, directory / "hidden_comparison.png")
        metadata = build_metadata(config, runner, kind="hidden")
        metadata["position_semantics"] = "Last token of each complete assistant-generation chat prefix; absolute positions may differ. These are not aligned question tokens. Final state includes final normalization."
        write_json(directory / "run_metadata.json", metadata)
        post_report = write_post_report(config, kind="hidden", directory=directory, rows=rows, metadata=metadata)
    except (Exception, KeyboardInterrupt) as error:
        failure = {"status": "failed", "error_type": type(error).__name__, "error": str(error), "effective_config": config}
        write_json(directory / "failure.json", failure)
        post_report = write_post_report(config, kind="hidden", directory=directory, rows=rows, metadata=metadata, error=f"{type(error).__name__}: {error}")
        print(f"Failure report: {post_report}", flush=True)
        raise
    print(f"Returned states: {runner.model.config.num_hidden_layers + 1}; hidden width: {runner.model.config.hidden_size}")
    print(f"Post-experiment PDF: {post_report}")
    print(f"Saved {len(rows)} comparison rows: {directory}")


if __name__ == "__main__":
    main()
