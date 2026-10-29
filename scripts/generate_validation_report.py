"""Generate expanded A/B validation PDFs offline from frozen inputs/results."""

import argparse
import json
from pathlib import Path

from truthgate.experiment import load_config
from truthgate.validation_reports import write_validation_post_report, write_validation_pre_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--phase", choices=("pre", "post"), required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("output/pdf/expanded_validation"))
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.phase == "pre":
        path = write_validation_pre_report(config, manifest=args.manifest, directory=args.output_dir)
    else:
        if args.run_dir is None:
            parser.error("--run-dir is required for post reports")
        def read(name, default):
            file = args.run_dir / name
            return json.loads(file.read_text(encoding="utf-8")) if file.is_file() else default
        metadata = read("run_metadata.json", {})
        config = metadata.get("effective_config", config)
        source = args.run_dir / "results.jsonl"
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()] if source.is_file() else []
        failure = read("failure.json", None)
        analysis = read("validation_summary.json", None)
        completion = read("completion.json", None)
        completed = bool(completion and completion.get("status") == "completed" and analysis)
        if completed and failure and failure.get("superseded_by_successful_resume"):
            failure = None
        path = write_validation_post_report(config, manifest=args.manifest, directory=args.output_dir,
                                            analysis=analysis, metadata=metadata, rows=rows,
                                            status="completed" if completed and failure is None else "incomplete",
                                            error=str(failure) if failure else None)
    print(path.resolve())


if __name__ == "__main__":
    main()
