"""Prepare a frozen PDF protocol, then separately execute the larger A/B baseline."""

import argparse
from pathlib import Path

from truthgate.experiment import load_config
from truthgate.validation_run import execute_validation, prepare_validation


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs/expanded.yaml")
    parser.add_argument("--manifest", type=Path, default=root / "data/expanded_dataset_manifest.json")
    parser.add_argument("--run-dir", type=Path, required=True)
    phase = parser.add_mutually_exclusive_group(required=True)
    phase.add_argument("--prepare", action="store_true")
    phase.add_argument("--execute", action="store_true")
    parser.add_argument("--pre-report-delivered", action="store_true",
                        help="Use only after supplying this run's frozen PDF to the user.")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.prepare:
        print(prepare_validation(config, args.manifest, args.run_dir), flush=True)
    else:
        execute_validation(config, args.manifest, args.run_dir,
                           pre_report_delivered=args.pre_report_delivered, resume=args.resume)


if __name__ == "__main__":
    main()
