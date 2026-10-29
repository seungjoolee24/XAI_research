"""Generate a research protocol or report offline from saved run files."""

import argparse
import json
from pathlib import Path

from truthgate.experiment import load_config
from truthgate.reports import KINDS, write_post_report, write_pre_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/baseline.yaml")
    parser.add_argument("--kind", required=True, choices=KINDS)
    parser.add_argument("--phase", required=True, choices=("pre", "post"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/pdf"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--include-swapped", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--prompt")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.phase == "pre":
        if args.limit is not None:
            if args.limit < 1 or (args.kind == "hidden" and args.limit > 3):
                parser.error("--limit must be positive and at most 3 for hidden")
            section, name = ("hidden_states", "question_limit") if args.kind == "hidden" else ("experiment", "limit")
            config[section][name] = args.limit
        if args.include_swapped:
            config["dataset"]["include_swapped"] = True
        if args.device:
            config["model"]["device"] = args.device
        path = write_pre_report(config, kind=args.kind, directory=args.output_dir, prompt=args.prompt)
    else:
        if args.run_dir is None:
            parser.error("--run-dir is required for a post-report")
        metadata_path = args.run_dir / "run_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
        config = metadata.get("effective_config", config)
        error_path = args.run_dir / "failure.json"
        if not error_path.is_file():
            error_path = args.run_dir / "error.json"
        error = json.loads(error_path.read_text(encoding="utf-8")) if error_path.is_file() else None
        if error is not None and isinstance(error, dict):
            if not metadata:
                config = error.get("effective_config", config)
            error = ": ".join(str(error[key]) for key in ("error_type", "error") if error.get(key)) or str(error)
        rows, result = None, None
        if error is None:
            if args.kind == "smoke":
                result = json.loads((args.run_dir / "result.json").read_text(encoding="utf-8"))
            else:
                filename = "hidden_states.jsonl" if args.kind == "hidden" else "results.jsonl"
                rows = [json.loads(line) for line in (args.run_dir / filename).read_text(encoding="utf-8").splitlines() if line.strip()]
        path = write_post_report(config, kind=args.kind, directory=args.output_dir, rows=rows,
                                 result=result, metadata=metadata, error=error)
    print(path.resolve())


if __name__ == "__main__":
    main()
