from __future__ import annotations

import argparse
from pathlib import Path

from .data import load_study_data, prepare_study_data, synthetic_listings
from .experiment import CoverageShiftExperiment, ExperimentConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coverage-shift",
        description="Run the vehicle asking-price training-coverage study.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--data",
        type=Path,
        help="processed vehicle listing CSV, for example vehicle_listings_processed.csv",
    )
    source.add_argument(
        "--synthetic-smoke",
        action="store_true",
        help="run the two-fold smoke test on deterministic synthetic listings",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory for checkpoints, predictions, and summary tables",
    )
    parser.add_argument(
        "--reference-year",
        type=int,
        default=2026,
        help="fixed year used to derive vehicle age (default: 2026)",
    )
    parser.add_argument(
        "--stage",
        choices=("smoke", "main", "paired", "sensitivity", "all"),
        help=(
            "experiment stage; main includes primary and alternate-boundary holdouts, "
            "while sensitivity runs the four modified-pipeline checks"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="refit models instead of reusing compatible checkpoints",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.synthetic_smoke:
        if args.stage not in {None, "smoke"}:
            parser.error("--synthetic-smoke can only run the smoke stage")
        frame = synthetic_listings(reference_year=args.reference_year)
        data = prepare_study_data(frame, reference_year=args.reference_year)
        stage = "smoke"
    else:
        if not args.data.is_file():
            parser.error(f"data file not found: {args.data}")
        data = load_study_data(args.data, reference_year=args.reference_year)
        stage = args.stage or "all"

    config = ExperimentConfig(
        output_dir=args.output_dir,
        reference_year=args.reference_year,
        force=args.force,
    )
    CoverageShiftExperiment(data, config).run(stage)


if __name__ == "__main__":
    main()
