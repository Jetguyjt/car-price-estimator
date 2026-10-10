#!/usr/bin/env python3
"""Publish aggregate, non-row-level artifacts from one completed full run.

This script does not fit models. It validates a completed experiment directory,
derives the reviewed public tables and summary, rebuilds the figures, and writes a
checksum manifest. Outputs are assembled in a staging directory first so a failed
validation does not leave a partially updated publication directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_RUN_FILES = (
    "data_manifest.json",
    "main_fold_metrics.csv",
    "main_oof_predictions.csv.gz",
    "main_summary.csv",
    "paired_range_fold_metrics.csv",
    "paired_range_summary.csv",
    "segment_rollup_min20.csv",
    "segment_summary.csv",
    "sensitivity_summary.csv",
)

PRIMARY_PROTOCOLS = {
    "random_reference": "Random listing-level reference",
    "holdout_mileage_primary": "Complete mileage-range holdout",
    "holdout_age_primary": "Complete vehicle-age-range holdout",
    "holdout_make_model": "Processed make/model-group holdout",
    "holdout_make": "Manufacturer-group holdout",
}

SUMMARY_CONDITIONS = {
    "random_reference": "random_listing_reference",
    "holdout_mileage_primary": "complete_mileage_ranges_withheld",
    "holdout_age_primary": "complete_vehicle_age_ranges_withheld",
    "holdout_make_model": "processed_make_model_groups_withheld",
    "holdout_make": "manufacturer_groups_withheld",
}

DESIGN_ROWS = (
    {
        "protocol": "random_reference",
        "evaluation_condition": "Random listing-level reference",
        "holdout_unit": "Individual listings",
        "cycles": 3,
        "fits": 15,
        "predictions_per_listing": 3,
    },
    {
        "protocol": "holdout_make_model",
        "evaluation_condition": "Processed make/model-group holdout",
        "holdout_unit": "Complete processed make/model groups",
        "cycles": 3,
        "fits": 15,
        "predictions_per_listing": 3,
    },
    {
        "protocol": "holdout_make",
        "evaluation_condition": "Manufacturer-group holdout",
        "holdout_unit": "Complete manufacturer groups",
        "cycles": 3,
        "fits": 15,
        "predictions_per_listing": 3,
    },
    {
        "protocol": "holdout_mileage_primary",
        "evaluation_condition": "Complete mileage-range holdout",
        "holdout_unit": "One of five fixed mileage ranges",
        "cycles": 1,
        "fits": 5,
        "predictions_per_listing": 1,
    },
    {
        "protocol": "holdout_age_primary",
        "evaluation_condition": "Complete vehicle-age-range holdout",
        "holdout_unit": "One of five fixed age ranges",
        "cycles": 1,
        "fits": 5,
        "predictions_per_listing": 1,
    },
)

PRIMARY_COLUMNS = (
    "protocol",
    "evaluation_condition",
    "n",
    "mae",
    "mae_delta",
    "mae_ratio",
    "delta_ci_low",
    "delta_ci_high",
    "ratio_ci_low",
    "ratio_ci_high",
    "mape_percent",
    "mdape_percent",
    "wape_percent",
    "rmse",
    "r2",
)

PAIRED_RANGES = (
    ("mileage", "35,000-59,999"),
    ("mileage", "100,000+"),
    ("car_age", "4-5"),
    ("car_age", "9+"),
)

BAND_ORDER = {
    "holdout_mileage_primary": (
        "mileage_primary",
        ("0-14,999", "15,000-34,999", "35,000-59,999", "60,000-99,999", "100,000+"),
    ),
    "holdout_age_primary": (
        "age_primary",
        ("0-1", "2-3", "4-5", "6-8", "9+"),
    ),
}

PUBLIC_SCOPES = {
    "figures/evaluation_design_schematic.png": (
        "figure",
        "Public evaluation-design schematic",
    ),
    "figures/matched_range_comparison.png": (
        "figure",
        "Public matched-range comparison",
    ),
    "figures/primary_ratio_forest.png": (
        "figure",
        "Public primary MAE-ratio comparison",
    ),
    "frozen_run_manifest.json": (
        "metadata",
        "Sanitized frozen-run fingerprint and settings",
    ),
    "summary.json": ("summary", "Aggregate study summary"),
    "tables/categorical_breadth.csv": (
        "table",
        "Aggregate categorical breadth by minimum group size",
    ),
    "tables/category_weighting_min20.csv": (
        "table",
        "Aggregate category weighting for groups with at least 20 records",
    ),
    "tables/dataset_summary.csv": ("table", "Aggregate sample summary"),
    "tables/evaluation_design.csv": (
        "table",
        "Aggregate evaluation design and fold sizes",
    ),
    "tables/matched_ranges.csv": ("table", "Aggregate matched-range results"),
    "tables/matching_diagnostics.csv": (
        "table",
        "Aggregate matched-composition diagnostics",
    ),
    "tables/numerical_band_direction.csv": (
        "table",
        "Aggregate signed-error direction by numerical band",
    ),
    "tables/numerical_band_results.csv": (
        "table",
        "Aggregate cohort results by numerical band",
    ),
    "tables/primary_protocols.csv": ("table", "Aggregate primary protocol metrics"),
    "tables/repetition_stability.csv": (
        "table",
        "Aggregate repeated-protocol stability",
    ),
    "tables/sensitivity_checks.csv": (
        "table",
        "Aggregate planned sensitivity checks",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_columns(frame: pd.DataFrame, source: str, columns: Iterable[str]) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(missing)}")


def _require_unique(frame: pd.DataFrame, source: str, columns: list[str]) -> None:
    if frame.duplicated(columns).any():
        raise ValueError(f"{source} has duplicate keys for {columns}")


def _require_rows(frame: pd.DataFrame, source: str) -> None:
    if frame.empty:
        raise ValueError(f"{source} is empty")


def _require_finite_public_table(frame: pd.DataFrame, name: str) -> None:
    _require_rows(frame, name)
    if frame.isna().any().any():
        raise ValueError(f"{name} contains missing values")
    numeric = frame.select_dtypes(include=[np.number])
    if numeric.size and not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"{name} contains non-finite values")


def _read_run_csv(run_dir: Path, name: str) -> pd.DataFrame:
    path = run_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"completed run is missing required file: {path}")
    frame = pd.read_csv(path)
    _require_rows(frame, name)
    return frame


def _select_indexed_rows(
    frame: pd.DataFrame,
    *,
    source: str,
    key: str,
    values: Iterable[str],
) -> pd.DataFrame:
    _require_columns(frame, source, [key])
    _require_unique(frame, source, [key])
    indexed = frame.set_index(key)
    values = list(values)
    missing = [value for value in values if value not in indexed.index]
    if missing:
        raise ValueError(f"{source} is missing registered {key} values: {missing}")
    return indexed.loc[values].reset_index()


def _validate_source_data(
    run_manifest: dict,
    raw: pd.DataFrame,
    clean: pd.DataFrame,
    processed: pd.DataFrame,
    processed_path: Path,
) -> None:
    required_manifest = {"source_sha256", "n_rows", "n_make_models", "n_makes"}
    missing = sorted(required_manifest.difference(run_manifest))
    if missing:
        raise ValueError(f"data_manifest.json is missing keys: {', '.join(missing)}")

    observed_sha = sha256(processed_path)
    if observed_sha != str(run_manifest["source_sha256"]):
        raise ValueError(
            "processed-data SHA-256 does not match the completed run manifest: "
            f"{observed_sha} != {run_manifest['source_sha256']}"
        )

    expected_rows = int(run_manifest["n_rows"])
    if len(processed) != expected_rows:
        raise ValueError(
            "processed-data row count does not match the completed run manifest: "
            f"{len(processed)} != {expected_rows}"
        )
    if not (len(raw) >= len(clean) >= len(processed)):
        raise ValueError(
            "expected raw rows >= clean rows >= processed rows, got "
            f"{len(raw)} >= {len(clean)} >= {len(processed)}"
        )
    if processed["make_model"].nunique() != int(run_manifest["n_make_models"]):
        raise ValueError("processed make/model count does not match data_manifest.json")
    if processed["make"].nunique() != int(run_manifest["n_makes"]):
        raise ValueError(
            "processed manufacturer count does not match data_manifest.json"
        )


def _dataset_summary(
    raw: pd.DataFrame, clean: pd.DataFrame, processed: pd.DataFrame
) -> pd.DataFrame:
    _require_columns(raw, "raw data", ["listing_key", "stock_type"])
    _require_columns(processed, "processed data", [
        "listing_key",
        "is_new",
        "make_model",
        "make",
        "price",
        "mileage",
        "year",
    ])
    _require_unique(raw, "raw data", ["listing_key"])
    _require_unique(processed, "processed data", ["listing_key"])

    stock_labels = processed[["listing_key", "is_new"]].merge(
        raw[["listing_key", "stock_type"]],
        on="listing_key",
        how="left",
        validate="one_to_one",
    )
    stock_counts = stock_labels["stock_type"].value_counts(dropna=False)
    group_sizes = processed.groupby("make_model", dropna=False).size()
    return pd.DataFrame(
        [
            ("Raw listing records", len(raw)),
            ("Records after primary cleaning", len(clean)),
            ("Model-ready records", len(processed)),
            ("Source-labeled used listings", int(stock_counts.get("Used", 0))),
            (
                "Source-labeled certified listings",
                int(stock_counts.get("Certified", 0)),
            ),
            ("Source-labeled new listings", int(stock_counts.get("New", 0))),
            ("Listings encoded non-new", int((processed["is_new"] == 0).sum())),
            ("Listings encoded new", int((processed["is_new"] == 1).sum())),
            ("Processed make/model categories", int(processed["make_model"].nunique())),
            ("Manufacturers", int(processed["make"].nunique())),
            ("Median asking price", float(processed["price"].median())),
            (
                "Asking-price interquartile range, low",
                float(processed["price"].quantile(0.25)),
            ),
            (
                "Asking-price interquartile range, high",
                float(processed["price"].quantile(0.75)),
            ),
            ("Asking-price range, low", float(processed["price"].min())),
            ("Asking-price range, high", float(processed["price"].max())),
            ("Median mileage", float(processed["mileage"].median())),
            (
                "Mileage interquartile range, low",
                float(processed["mileage"].quantile(0.25)),
            ),
            (
                "Mileage interquartile range, high",
                float(processed["mileage"].quantile(0.75)),
            ),
            ("Vehicle-year range, low", int(processed["year"].min())),
            ("Vehicle-year range, high", int(processed["year"].max())),
            ("Median processed-group size", float(group_sizes.median())),
            ("Singleton processed groups", int((group_sizes == 1).sum())),
            (
                "Processed groups with fewer than five listings",
                int((group_sizes < 5).sum()),
            ),
        ],
        columns=["quantity", "value"],
    )


def _evaluation_design(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    _require_columns(fold_metrics, "main_fold_metrics.csv", [
        "protocol",
        "fold",
        "train_n",
        "test_n",
    ])
    design = pd.DataFrame(DESIGN_ROWS)
    size_summary = fold_metrics.groupby("protocol", as_index=False).agg(
        observed_fits=("fold", "size"),
        train_rows_min=("train_n", "min"),
        train_rows_max=("train_n", "max"),
        test_rows_min=("test_n", "min"),
        test_rows_max=("test_n", "max"),
    )
    design = design.merge(
        size_summary, on="protocol", how="left", validate="one_to_one"
    )
    if design["observed_fits"].isna().any():
        missing = design.loc[design["observed_fits"].isna(), "protocol"].tolist()
        raise ValueError(f"main_fold_metrics.csv is missing protocols: {missing}")
    if not (design["fits"] == design["observed_fits"]).all():
        mismatch = design.loc[
            design["fits"] != design["observed_fits"],
            ["protocol", "fits", "observed_fits"],
        ].to_dict("records")
        raise ValueError(f"registered fit counts do not match fold records: {mismatch}")
    if (design[["train_rows_min", "test_rows_min"]] <= 0).any().any():
        raise ValueError("main fold records contain an empty train or test split")
    return design


def _primary_protocols(main_summary: pd.DataFrame) -> pd.DataFrame:
    ordered = _select_indexed_rows(
        main_summary,
        source="main_summary.csv",
        key="protocol",
        values=PRIMARY_PROTOCOLS,
    )
    ordered.insert(
        1, "evaluation_condition", ordered["protocol"].map(PRIMARY_PROTOCOLS)
    )
    _require_columns(ordered, "main_summary.csv", PRIMARY_COLUMNS)
    return ordered[list(PRIMARY_COLUMNS)]


def _matched_ranges(paired: pd.DataFrame) -> pd.DataFrame:
    _require_columns(paired, "paired_range_summary.csv", ["feature", "band"])
    _require_unique(paired, "paired_range_summary.csv", ["feature", "band"])
    observed = set(zip(paired["feature"].astype(str), paired["band"].astype(str)))
    missing = [key for key in PAIRED_RANGES if key not in observed]
    if missing:
        raise ValueError(
            f"paired_range_summary.csv is missing registered ranges: {missing}"
        )
    rank = {key: index for index, key in enumerate(PAIRED_RANGES)}
    result = paired.copy()
    result["_publish_order"] = [
        rank.get((str(row.feature), str(row.band)), len(rank))
        for row in result.itertuples()
    ]
    result = result.sort_values("_publish_order").drop(columns="_publish_order")
    return result.reset_index(drop=True)


def _sensitivity_checks(
    main_summary: pd.DataFrame, sensitivity: pd.DataFrame
) -> pd.DataFrame:
    main = _select_indexed_rows(
        main_summary,
        source="main_summary.csv",
        key="protocol",
        values=("holdout_mileage_sensitivity", "holdout_age_sensitivity"),
    ).set_index("protocol")
    _require_columns(sensitivity, "sensitivity_summary.csv", [
        "analysis_set",
        "protocol",
        "mae",
        "mae_ratio",
    ])
    _require_unique(
        sensitivity, "sensitivity_summary.csv", ["analysis_set", "protocol"]
    )
    sensitivity = sensitivity.set_index(["analysis_set", "protocol"])

    def modified_row(
        label: str, condition: str, analysis_set: str, protocol: str
    ) -> dict[str, object]:
        shifted_key = (analysis_set, protocol)
        reference_key = (analysis_set, "random_reference")
        missing = [
            key
            for key in (shifted_key, reference_key)
            if key not in sensitivity.index
        ]
        if missing:
            raise ValueError(f"sensitivity_summary.csv is missing rows: {missing}")
        shifted = sensitivity.loc[shifted_key]
        reference = sensitivity.loc[reference_key]
        return {
            "modified_analysis": label,
            "random_reference_mae": float(reference["mae"]),
            "shifted_condition": condition,
            "shifted_mae": float(shifted["mae"]),
            "ratio_to_own_reference": float(shifted["mae_ratio"]),
        }

    rows = []
    for protocol, condition in (
        ("holdout_mileage_sensitivity", "Mileage ranges"),
        ("holdout_age_sensitivity", "Age ranges"),
    ):
        shifted = main.loc[protocol]
        rows.append({
            "modified_analysis": "Alternate boundaries",
            "random_reference_mae": float(shifted["reference_mae"]),
            "shifted_condition": condition,
            "shifted_mae": float(shifted["mae"]),
            "ratio_to_own_reference": float(shifted["mae_ratio"]),
        })
    rows.extend([
        modified_row(
            "Remove make/model price encoding",
            "Make/model groups",
            "no_price_encoding",
            "holdout_make_model",
        ),
        modified_row(
            "Remove make/model price encoding",
            "Manufacturers",
            "no_price_encoding",
            "holdout_make",
        ),
        modified_row(
            "Remove mileage bucket and miles/year",
            "Mileage ranges",
            "mileage_proxy_reduced",
            "holdout_mileage_primary",
        ),
        modified_row(
            "Remove 25 exact-content later copies",
            "Make/model groups",
            "strict_exact_content_dedup",
            "holdout_make_model",
        ),
        modified_row(
            "Remove 25 exact-content later copies",
            "Manufacturers",
            "strict_exact_content_dedup",
            "holdout_make",
        ),
        modified_row(
            "Add manufacturer as a model input",
            "Make/model groups",
            "manufacturer_aware_model",
            "holdout_make_model",
        ),
        modified_row(
            "Add manufacturer as a model input",
            "Manufacturers",
            "manufacturer_aware_model",
            "holdout_make",
        ),
    ])
    return pd.DataFrame(rows)


def _numerical_band_results(segments: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        segments,
        "segment_summary.csv",
        ["protocol", "segment_variable", "segment"],
    )
    result = segments[
        segments["protocol"].isin(BAND_ORDER)
    ].copy()
    expected = {
        (protocol, band)
        for protocol, (_, bands) in BAND_ORDER.items()
        for band in bands
    }
    observed = set(zip(result["protocol"].astype(str), result["segment"].astype(str)))
    missing = sorted(expected.difference(observed))
    if missing:
        raise ValueError(f"segment_summary.csv is missing numerical bands: {missing}")
    return result


def _repetition_stability(predictions: pd.DataFrame) -> pd.DataFrame:
    _require_columns(predictions, "main_oof_predictions.csv.gz", [
        "protocol",
        "repetition",
        "abs_error",
        "ape",
    ])
    repeated = predictions[predictions["protocol"].isin(
        ("random_reference", "holdout_make_model", "holdout_make")
    )]
    return repeated.groupby(["protocol", "repetition"], as_index=False).agg(
        mae=("abs_error", "mean"),
        mape_percent=("ape", lambda values: 100 * values.mean()),
    )


def _categorical_breadth(segments: pd.DataFrame) -> pd.DataFrame:
    _require_columns(segments, "segment_summary.csv", [
        "protocol",
        "segment_variable",
        "n",
        "mae_delta",
    ])
    rows = []
    for protocol, variable in (
        ("holdout_make_model", "make_model"),
        ("holdout_make", "make"),
    ):
        subset = segments[
            (segments["protocol"] == protocol)
            & (segments["segment_variable"] == variable)
        ]
        if subset.empty:
            raise ValueError(
                f"segment_summary.csv has no {protocol}/{variable} segment rows"
            )
        for minimum_n in (1, 20, 50):
            eligible = subset[subset["n"] >= minimum_n]
            if eligible.empty:
                raise ValueError(
                    f"no {protocol}/{variable} groups meet minimum n={minimum_n}"
                )
            rows.append({
                "protocol": protocol,
                "minimum_group_n": minimum_n,
                "eligible_groups": int(len(eligible)),
                "eligible_rows": int(eligible["n"].sum()),
                "groups_with_higher_mae": int((eligible["mae_delta"] > 0).sum()),
                "fraction_with_higher_mae": float((eligible["mae_delta"] > 0).mean()),
            })
    return pd.DataFrame(rows)


def _category_weighting(group_rollups: pd.DataFrame) -> pd.DataFrame:
    _require_columns(group_rollups, "segment_rollup_min20.csv", ["protocol"])
    result = group_rollups[group_rollups["protocol"].isin(
        ("holdout_make_model", "holdout_make")
    )].copy()
    if set(result["protocol"]) != {"holdout_make_model", "holdout_make"}:
        raise ValueError("segment_rollup_min20.csv is missing a categorical protocol")
    return result


def _matching_diagnostics(paired_folds: pd.DataFrame) -> pd.DataFrame:
    _require_columns(paired_folds, "paired_range_fold_metrics.csv", [
        "feature",
        "band",
        "condition",
        "same_make_model_match_rate",
        "same_make_or_model_match_rate",
        "unrestricted_fallback_matches",
    ])
    represented = paired_folds[paired_folds["condition"] == "range_represented"]
    if represented.empty:
        raise ValueError(
            "paired_range_fold_metrics.csv has no represented-condition rows"
        )
    return represented.groupby(["feature", "band"], as_index=False).agg(
        mean_exact_make_model_match_rate=("same_make_model_match_rate", "mean"),
        mean_make_or_model_match_rate=("same_make_or_model_match_rate", "mean"),
        unrestricted_fallback_matches=("unrestricted_fallback_matches", "sum"),
    )


def _numerical_band_direction(predictions: pd.DataFrame) -> pd.DataFrame:
    _require_columns(predictions, "main_oof_predictions.csv.gz", [
        "row_id",
        "protocol",
        "actual",
        "prediction",
        "error",
        "mileage_primary",
        "age_primary",
    ])
    rows = []
    for protocol, (band_column, bands) in BAND_ORDER.items():
        frame = predictions[predictions["protocol"] == protocol]
        if frame.empty:
            raise ValueError(f"main_oof_predictions.csv.gz has no rows for {protocol}")
        per_row = frame.groupby(["row_id", band_column], as_index=False).agg(
            actual=("actual", "first"),
            prediction=("prediction", "mean"),
            error=("error", "mean"),
        )
        for band in bands:
            group = per_row[per_row[band_column] == band]
            if group.empty:
                raise ValueError(f"{protocol} has no predictions for band {band}")
            rows.append({
                "protocol": protocol,
                "band": band,
                "n": int(len(group)),
                "share_overpredicted_percent": round(
                    float(100 * (group["prediction"] > group["actual"]).mean()), 1
                ),
                "mean_signed_error": float(group["error"].mean()),
            })
    return pd.DataFrame(rows)


def build_public_tables(
    *,
    run_dir: Path,
    raw_data: Path,
    clean_data: Path,
    processed_data: Path,
) -> dict[str, pd.DataFrame]:
    """Validate inputs and return every public table keyed by output filename."""
    run_dir = Path(run_dir)
    missing_files = [
        name for name in REQUIRED_RUN_FILES if not (run_dir / name).is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            "completed run is missing required files: " + ", ".join(missing_files)
        )
    for path, label in (
        (raw_data, "raw data"),
        (clean_data, "clean data"),
        (processed_data, "processed data"),
    ):
        if not Path(path).is_file():
            raise FileNotFoundError(f"{label} file not found: {path}")

    run_manifest = json.loads((run_dir / "data_manifest.json").read_text())
    raw = pd.read_csv(raw_data)
    clean = pd.read_csv(clean_data)
    processed = pd.read_csv(processed_data)
    _require_rows(raw, "raw data")
    _require_rows(clean, "clean data")
    _require_rows(processed, "processed data")
    _require_columns(processed, "processed data", ["make_model", "make"])
    _validate_source_data(run_manifest, raw, clean, processed, Path(processed_data))

    main_folds = _read_run_csv(run_dir, "main_fold_metrics.csv")
    main_predictions = _read_run_csv(run_dir, "main_oof_predictions.csv.gz")
    main_summary = _read_run_csv(run_dir, "main_summary.csv")
    paired_folds = _read_run_csv(run_dir, "paired_range_fold_metrics.csv")
    paired = _read_run_csv(run_dir, "paired_range_summary.csv")
    group_rollups = _read_run_csv(run_dir, "segment_rollup_min20.csv")
    segments = _read_run_csv(run_dir, "segment_summary.csv")
    sensitivity = _read_run_csv(run_dir, "sensitivity_summary.csv")

    tables = {
        "dataset_summary.csv": _dataset_summary(raw, clean, processed),
        "evaluation_design.csv": _evaluation_design(main_folds),
        "primary_protocols.csv": _primary_protocols(main_summary),
        "matched_ranges.csv": _matched_ranges(paired),
        "sensitivity_checks.csv": _sensitivity_checks(main_summary, sensitivity),
        "numerical_band_results.csv": _numerical_band_results(segments),
        "repetition_stability.csv": _repetition_stability(main_predictions),
        "categorical_breadth.csv": _categorical_breadth(segments),
        "category_weighting_min20.csv": _category_weighting(group_rollups),
        "matching_diagnostics.csv": _matching_diagnostics(paired_folds),
        "numerical_band_direction.csv": _numerical_band_direction(main_predictions),
    }
    for name, frame in tables.items():
        _require_finite_public_table(frame, name)
    return tables


def build_summary(tables: dict[str, pd.DataFrame]) -> dict:
    dataset = tables["dataset_summary.csv"].set_index("quantity")["value"]
    primary = tables["primary_protocols.csv"].set_index("protocol")
    matched = tables["matched_ranges.csv"]

    primary_records = []
    for protocol, condition in SUMMARY_CONDITIONS.items():
        row = primary.loc[protocol]
        record = {
            "condition": condition,
            "mae_usd": float(row["mae"]),
            "mape_percent": float(row["mape_percent"]),
            "mae_ratio": float(row["mae_ratio"]),
        }
        if protocol != "random_reference":
            record["mae_ratio_interval_95"] = [
                float(row["ratio_ci_low"]),
                float(row["ratio_ci_high"]),
            ]
        primary_records.append(record)

    matched_records = []
    for row in matched.itertuples(index=False):
        feature = "vehicle_age" if row.feature == "car_age" else str(row.feature)
        band = str(row.band).replace(",", "")
        matched_records.append({
            "feature": feature,
            "range": band,
            "region": str(row.region_type),
            "records": int(row.n),
            "represented_mae_usd": float(row.represented_mae),
            "absent_mae_usd": float(row.absent_mae),
            "mae_ratio": float(row.mae_ratio),
        })

    return {
        "study_sample": {
            "raw_records": int(dataset["Raw listing records"]),
            "records_after_primary_cleaning": int(
                dataset["Records after primary cleaning"]
            ),
            "model_ready_records": int(dataset["Model-ready records"]),
            "processed_make_model_categories": int(
                dataset["Processed make/model categories"]
            ),
            "manufacturers": int(dataset["Manufacturers"]),
        },
        "primary_protocols": primary_records,
        "matched_ranges": matched_records,
        "scope_note": (
            "Aggregate results for one fixed model pipeline and one nonrandom "
            "marketplace sample. Asking prices are not completed-sale prices."
        ),
    }


def _ordered_band_counts(band_counts: dict) -> dict[str, dict[str, int]]:
    orders = {
        "age_primary": ("0-1", "2-3", "4-5", "6-8", "9+"),
        "age_sensitivity": ("0", "1-2", "3-4", "5-8", "9+"),
        "mileage_primary": (
            "0-14,999",
            "15,000-34,999",
            "35,000-59,999",
            "60,000-99,999",
            "100,000+",
        ),
        "mileage_sensitivity": (
            "0-9,999",
            "10,000-24,999",
            "25,000-49,999",
            "50,000-99,999",
            "100,000+",
        ),
    }
    missing_specs = sorted(set(orders).difference(band_counts))
    if missing_specs:
        raise ValueError(
            "data_manifest.json is missing registered band counts: "
            + ", ".join(missing_specs)
        )
    result = {}
    for spec, labels in orders.items():
        missing_labels = [label for label in labels if label not in band_counts[spec]]
        if missing_labels:
            raise ValueError(
                f"data_manifest.json is missing {spec} labels: {missing_labels}"
            )
        result[spec] = {label: int(band_counts[spec][label]) for label in labels}
    return result


def _sanitize_frozen_run_manifest(run_manifest: dict) -> dict:
    required = {
        "source_sha256",
        "experiment_code_version",
        "experiment_code_sha256",
        "reference_year",
        "n_rows",
        "n_input_features_before_price_encoding",
        "feature_order",
        "categorical_features",
        "model_parameters",
        "monotone_directions",
        "target",
        "n_make_models",
        "n_makes",
        "price_min",
        "price_max",
        "exact_content_duplicate_rows_after_first",
        "versions",
        "band_counts",
    }
    missing = sorted(required.difference(run_manifest))
    if missing:
        raise ValueError(
            "data_manifest.json cannot produce frozen_run_manifest.json; "
            "missing keys: "
            + ", ".join(missing)
            + ". Supply --frozen-run-manifest for a validated legacy run."
        )
    monotone = run_manifest["monotone_directions"]
    ordered_monotone = {
        name: int(monotone[name]) for name in ("year", "mileage", "owner_count")
    }
    model_parameter_order = (
        "n_estimators",
        "learning_rate",
        "num_leaves",
        "min_child_samples",
        "feature_fraction",
        "bagging_fraction",
        "bagging_freq",
        "random_state",
        "n_jobs",
        "verbose",
        "objective",
    )
    model_parameters = run_manifest["model_parameters"]
    missing_parameters = [
        name for name in model_parameter_order if name not in model_parameters
    ]
    if missing_parameters:
        raise ValueError(
            "data_manifest.json is missing model parameters: "
            + ", ".join(missing_parameters)
        )
    ordered_parameters = {
        name: model_parameters[name] for name in model_parameter_order
    }
    version_order = ("python", "numpy", "pandas", "scikit_learn", "lightgbm")
    versions = run_manifest["versions"]
    missing_versions = [name for name in version_order if name not in versions]
    if missing_versions:
        raise ValueError(
            "data_manifest.json is missing runtime versions: "
            + ", ".join(missing_versions)
        )
    ordered_versions = {name: str(versions[name]) for name in version_order}
    return {
        "source_file": "private processed snapshot",
        "source_sha256": str(run_manifest["source_sha256"]),
        "experiment_code_version": str(run_manifest["experiment_code_version"]),
        "experiment_code_sha256": str(run_manifest["experiment_code_sha256"]),
        "reference_year": int(run_manifest["reference_year"]),
        "n_rows": int(run_manifest["n_rows"]),
        "n_input_features_before_price_encoding": int(
            run_manifest["n_input_features_before_price_encoding"]
        ),
        "feature_order": list(run_manifest["feature_order"]),
        "categorical_features": list(run_manifest["categorical_features"]),
        "model_parameters": ordered_parameters,
        "monotone_directions": ordered_monotone,
        "target": str(run_manifest["target"]),
        "n_make_models": int(run_manifest["n_make_models"]),
        "n_makes": int(run_manifest["n_makes"]),
        "price_min": float(run_manifest["price_min"]),
        "price_max": float(run_manifest["price_max"]),
        "exact_content_duplicate_rows_after_first": int(
            run_manifest["exact_content_duplicate_rows_after_first"]
        ),
        "versions": ordered_versions,
        "band_counts": _ordered_band_counts(run_manifest["band_counts"]),
    }


def _validate_supplied_frozen_manifest(
    frozen: dict,
    run_manifest: dict,
    processed: pd.DataFrame,
    processed_path: Path,
    source: Path,
) -> None:
    required = {
        "source_file",
        "source_sha256",
        "experiment_code_version",
        "experiment_code_sha256",
        "reference_year",
        "n_rows",
        "n_input_features_before_price_encoding",
        "feature_order",
        "categorical_features",
        "model_parameters",
        "monotone_directions",
        "target",
        "n_make_models",
        "n_makes",
        "price_min",
        "price_max",
        "exact_content_duplicate_rows_after_first",
        "versions",
        "band_counts",
    }
    missing = sorted(required.difference(frozen))
    if missing:
        raise ValueError(
            f"{source} is missing frozen-run metadata: {', '.join(missing)}"
        )
    if frozen["source_file"] != "private processed snapshot":
        raise ValueError(
            "the supplied frozen-run manifest must replace its source path with "
            "'private processed snapshot'"
        )
    expected = {
        "source_sha256": sha256(processed_path),
        "n_rows": len(processed),
        "n_make_models": processed["make_model"].nunique(),
        "n_makes": processed["make"].nunique(),
    }
    for key, value in expected.items():
        if frozen[key] != value:
            raise ValueError(
                f"{source} {key} does not match the completed run and source data"
            )
    for key in required.difference({"source_file"}):
        if key in run_manifest and frozen[key] != run_manifest[key]:
            raise ValueError(f"{source} {key} does not match data_manifest.json")


def build_frozen_run_manifest(
    *,
    run_manifest: dict,
    processed: pd.DataFrame,
    processed_path: Path,
    supplied_manifest: Path | None,
) -> tuple[dict, bytes | None]:
    """Return validated public run metadata and optional original bytes to preserve."""
    if supplied_manifest is None:
        return _sanitize_frozen_run_manifest(run_manifest), None
    source = Path(supplied_manifest)
    if not source.is_file():
        raise FileNotFoundError(f"frozen-run manifest file not found: {source}")
    source_bytes = source.read_bytes()
    frozen = json.loads(source_bytes)
    _validate_supplied_frozen_manifest(
        frozen, run_manifest, processed, processed_path, source
    )
    return frozen, source_bytes


def _render_figures(tables_dir: Path, output_dir: Path) -> None:
    try:
        from . import make_figures
    except ImportError:  # direct execution: python scripts/publish_results.py
        import make_figures

    make_figures.apply_style()
    make_figures.evaluation_design_schematic(output_dir)
    make_figures.primary_ratio_forest(tables_dir, output_dir)
    make_figures.matched_range_comparison(tables_dir, output_dir)


def _write_manifest(staging_dir: Path) -> None:
    rows = []
    for relative_path in sorted(PUBLIC_SCOPES):
        path = staging_dir / relative_path
        if not path.is_file():
            raise FileNotFoundError(
                f"publication artifact was not generated: {relative_path}"
            )
        artifact_type, scope = PUBLIC_SCOPES[relative_path]
        rows.append({
            "path": relative_path,
            "artifact_type": artifact_type,
            "scope": scope,
            "sha256": sha256(path),
        })
    with (staging_dir / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("path", "artifact_type", "scope", "sha256"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def publish_results(
    *,
    run_dir: Path,
    raw_data: Path,
    clean_data: Path,
    processed_data: Path,
    output_dir: Path,
    frozen_run_manifest: Path | None = None,
) -> list[Path]:
    """Build and validate the public artifacts before promoting generated files."""
    run_dir = Path(run_dir).resolve()
    raw_data = Path(raw_data).resolve()
    clean_data = Path(clean_data).resolve()
    processed_data = Path(processed_data).resolve()
    output_dir = Path(output_dir).resolve()
    tables = build_public_tables(
        run_dir=run_dir,
        raw_data=raw_data,
        clean_data=clean_data,
        processed_data=processed_data,
    )
    summary = build_summary(tables)
    run_manifest = json.loads((run_dir / "data_manifest.json").read_text())
    processed = pd.read_csv(processed_data)
    frozen, frozen_bytes = build_frozen_run_manifest(
        run_manifest=run_manifest,
        processed=processed,
        processed_path=processed_data,
        supplied_manifest=frozen_run_manifest,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".publish-results-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        tables_dir = staging / "tables"
        figures_dir = staging / "figures"
        tables_dir.mkdir(parents=True)
        for name, frame in tables.items():
            frame.to_csv(tables_dir / name, index=False)
        (staging / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n"
        )
        if frozen_bytes is None:
            (staging / "frozen_run_manifest.json").write_text(
                json.dumps(frozen, indent=2, allow_nan=False) + "\n"
            )
        else:
            (staging / "frozen_run_manifest.json").write_bytes(frozen_bytes)
        _render_figures(tables_dir, figures_dir)
        _write_manifest(staging)

        generated = [Path(path) for path in sorted(PUBLIC_SCOPES)] + [
            Path("manifest.csv")
        ]
        for relative_path in generated:
            source = staging / relative_path
            target = output_dir / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, target)
    return [output_dir / path for path in generated]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, required=True, help="completed full-run directory"
    )
    parser.add_argument("--raw-data", type=Path, required=True, help="raw listing CSV")
    parser.add_argument(
        "--clean-data", type=Path, required=True, help="cleaned listing CSV"
    )
    parser.add_argument(
        "--processed-data",
        type=Path,
        required=True,
        help="processed CSV used by the completed run",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="destination for public tables, summary, figures, and manifest",
    )
    parser.add_argument(
        "--frozen-run-manifest",
        type=Path,
        help=(
            "validated sanitized metadata to preserve for a legacy run whose "
            "data_manifest.json predates current provenance fields"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    paths = publish_results(
        run_dir=args.run_dir,
        raw_data=args.raw_data,
        clean_data=args.clean_data,
        processed_data=args.processed_data,
        output_dir=args.output_dir,
        frozen_run_manifest=args.frozen_run_manifest,
    )
    print(f"Published {len(paths) - 1} artifacts plus manifest to {args.output_dir}")


if __name__ == "__main__":
    main()
