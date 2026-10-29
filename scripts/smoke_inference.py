"""Run one Qwen3 prompt before a full experiment."""

import argparse
import json
from pathlib import Path

from truthgate.experiment import build_metadata, create_run_directory, load_config, write_json
from truthgate.model import CausalLMRunner, set_seed
from truthgate.reports import write_pre_report, write_post_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/baseline.yaml")
    parser.add_argument("--prompt", default="What is the most abundant gas in Earth's atmosphere? A. Oxygen B. Nitrogen. Answer only A or B.")
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.device:
        config["model"]["device"] = args.device
    directory = create_run_directory(config, "smoke", args.output_dir)
    pre_report = write_pre_report(config, kind="smoke", directory=directory, prompt=args.prompt)
    print(f"Pre-experiment PDF (before model load): {pre_report}", flush=True)
    try:
        set_seed(config["seed"])
        runner = CausalLMRunner.from_config(config["model"])
        result = runner.evaluate_prompt(args.prompt)
        write_json(directory / "result.json", result)
        metadata = build_metadata(config, runner, kind="smoke")
        metadata["smoke_prompt"] = args.prompt
        write_json(directory / "run_metadata.json", metadata)
        post_report = write_post_report(config, kind="smoke", directory=directory, result=result, metadata=metadata)
    except (Exception, KeyboardInterrupt) as error:
        failure = {"status": "failed", "error_type": type(error).__name__, "error": str(error), "effective_config": config}
        write_json(directory / "failure.json", failure)
        post_report = write_post_report(config, kind="smoke", directory=directory, error=f"{type(error).__name__}: {error}")
        print(f"Failure report: {post_report}", flush=True)
        raise
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    print(f"Post-experiment PDF: {post_report}")
    print(f"Saved smoke inference: {directory}")


if __name__ == "__main__":
    main()
