"""Validate the public aggregate results and their recorded checksums."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_close(actual: float, expected: float, label: str) -> None:
    if not np.isclose(actual, expected, rtol=1e-10, atol=1e-10):
        raise AssertionError(f"{label}: {actual} != {expected}")


def validate_manifest() -> int:
    manifest_path = RESULTS / "manifest.csv"
    with manifest_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise AssertionError("results manifest is empty")
    for row in rows:
        path = RESULTS / row["path"]
        if not path.is_file():
            raise AssertionError(f"manifest file is missing: {row['path']}")
        observed = sha256(path)
        if observed != row["sha256"]:
            raise AssertionError(f"checksum mismatch: {row['path']}")
    return len(rows)


def validate_primary_summary() -> None:
    summary = json.loads((RESULTS / "summary.json").read_text())
    table = pd.read_csv(RESULTS / "tables" / "primary_protocols.csv").set_index(
        "protocol"
    )
    reference_mae = float(table.loc["random_reference", "mae"])
    mapping = {
        "random_listing_reference": "random_reference",
        "complete_mileage_ranges_withheld": "holdout_mileage_primary",
        "complete_vehicle_age_ranges_withheld": "holdout_age_primary",
        "processed_make_model_groups_withheld": "holdout_make_model",
        "manufacturer_groups_withheld": "holdout_make",
    }
    for record in summary["primary_protocols"]:
        row = table.loc[mapping[record["condition"]]]
        require_close(record["mae_usd"], row["mae"], f"{record['condition']} MAE")
        require_close(
            record["mape_percent"], row["mape_percent"], f"{record['condition']} MAPE"
        )
        require_close(
            record["mae_ratio"], row["mae_ratio"], f"{record['condition']} MAE ratio"
        )
        require_close(row["mae_delta"], row["mae"] - reference_mae, "MAE delta")
        require_close(row["mae_ratio"], row["mae"] / reference_mae, "MAE ratio")
        if "mae_ratio_interval_95" in record:
            low, high = record["mae_ratio_interval_95"]
            require_close(low, row["ratio_ci_low"], "ratio interval lower bound")
            require_close(high, row["ratio_ci_high"], "ratio interval upper bound")


def validate_matched_summary() -> None:
    summary = json.loads((RESULTS / "summary.json").read_text())
    table = pd.read_csv(RESULTS / "tables" / "matched_ranges.csv")
    keys = {
        (str(row.feature), str(row.band)): row for row in table.itertuples(index=False)
    }
    for record in summary["matched_ranges"]:
        public_feature = (
            "car_age" if record["feature"] == "vehicle_age" else record["feature"]
        )
        band = (
            record["range"]
            .replace("35000-59999", "35,000-59,999")
            .replace("100000+", "100,000+")
        )
        row = keys[(public_feature, band)]
        require_close(
            record["represented_mae_usd"], row.represented_mae, "represented MAE"
        )
        require_close(record["absent_mae_usd"], row.absent_mae, "absent MAE")
        require_close(record["mae_ratio"], row.mae_ratio, "matched MAE ratio")
        require_close(
            row.mae_delta, row.absent_mae - row.represented_mae, "matched MAE delta"
        )
        require_close(
            row.mae_ratio, row.absent_mae / row.represented_mae, "matched ratio"
        )


def validate_design() -> None:
    design = pd.read_csv(RESULTS / "tables" / "evaluation_design.csv")
    if len(design) != 5:
        raise AssertionError("evaluation design must contain five primary conditions")
    if not (design["fits"] == design["observed_fits"]).all():
        raise AssertionError("reported and observed fit counts differ")
    if (design[["train_rows_min", "test_rows_min"]] <= 0).any().any():
        raise AssertionError(
            "evaluation design contains an empty training or test fold"
        )


def main() -> None:
    count = validate_manifest()
    validate_primary_summary()
    validate_matched_summary()
    validate_design()
    print(f"Validated {count} public artifacts and all headline result cross-checks.")


if __name__ == "__main__":
    main()
