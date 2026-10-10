from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .data import (
    BAND_SPECS,
    ENCODING_FEATURE,
    PAIRED_RANGES,
    PRIMARY_AGE,
    PRIMARY_MILEAGE,
    SENSITIVITY_AGE,
    SENSITIVITY_MILEAGE,
    StudyData,
)
from .metrics import metric_record, paired_cluster_bootstrap
from .model import (
    ENCODING_SMOOTHING,
    MODEL_PARAMETERS,
    MONOTONE_DIRECTIONS,
    fit_predict,
)
from .splits import (
    balanced_group_splits,
    band_splits,
    matched_outside_removals,
    random_splits,
    standard_splits,
)

EXPERIMENT_CODE_VERSION = "1.0.0"


@dataclass(frozen=True)
class ExperimentConfig:
    output_dir: Path
    reference_year: int = 2026
    n_splits: int = 5
    seeds: tuple[int, ...] = (42, 43, 44)
    bootstrap_reps: int = 2_000
    force: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.n_splits != 5:
            raise ValueError("the registered design uses five folds")
        if tuple(self.seeds) != (42, 43, 44):
            raise ValueError("the registered design uses seeds 42, 43, and 44")
        if self.bootstrap_reps < 1:
            raise ValueError("bootstrap_reps must be positive")


class CoverageShiftExperiment:
    def __init__(self, data: StudyData, config: ExperimentConfig):
        self.data = data
        self.config = config
        self.output_dir = config.output_dir
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.data_tag = _study_digest(data, reference_year=config.reference_year)[:16]

    def run(self, stage: str = "all") -> None:
        valid_stages = {"smoke", "main", "paired", "sensitivity", "all"}
        if stage not in valid_stages:
            raise ValueError(f"unknown stage: {stage}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.write_manifest()
        print(
            f"Loaded {len(self.data.features):,} rows; "
            f"{self.data.metadata['make_model'].nunique():,} make/model groups; "
            f"{self.data.metadata['make'].nunique():,} manufacturers.",
            flush=True,
        )
        if stage == "smoke":
            self.run_smoke()
            return
        if stage in {"main", "all"}:
            self.run_main()
        if stage in {"paired", "all"}:
            self.run_paired_ranges()
        if stage in {"sensitivity", "all"}:
            self.run_sensitivities()
        print(f"Results written to {self.output_dir}", flush=True)

    def write_manifest(self) -> None:
        source = self.data.source_path
        payload = {
            "source_file": source.name if source is not None else "synthetic",
            "source_sha256": _sha256(source) if source is not None else self.data_tag,
            "experiment_code_version": EXPERIMENT_CODE_VERSION,
            "experiment_code_sha256": _package_source_digest(),
            "reference_year": int(self.config.reference_year),
            "n_rows": int(len(self.data.features)),
            "n_input_features_before_price_encoding": int(self.data.features.shape[1]),
            "feature_order": list(self.data.feature_order),
            "categorical_features": list(self.data.categorical_features),
            "model_parameters": dict(MODEL_PARAMETERS),
            "monotone_directions": dict(MONOTONE_DIRECTIONS),
            "target": "log1p(price)",
            "n_make_models": int(self.data.metadata["make_model"].nunique()),
            "n_makes": int(self.data.metadata["make"].nunique()),
            "price_min": float(self.data.target.min()),
            "price_max": float(self.data.target.max()),
            "exact_content_duplicate_rows_after_first": int(
                self.data.metadata["content_fingerprint"].duplicated().sum()
            ),
            "versions": _runtime_versions(),
            "band_counts": {
                spec.name: {
                    label: int((self.data.metadata[spec.name] == label).sum())
                    for label in spec.labels
                }
                for spec in BAND_SPECS.values()
            },
        }
        _json_dump(self.output_dir / "data_manifest.json", payload)

    def _checkpoint_paths(self, key: str) -> tuple[Path, Path]:
        safe_key = _slug(key)
        stem = f"{self.data_tag}__{safe_key}"
        return (
            self.checkpoint_dir / f"{stem}.npz",
            self.checkpoint_dir / f"{stem}.folds.json",
        )

    def _run_oof(
        self,
        *,
        key: str,
        protocol: str,
        repetition: int,
        features: pd.DataFrame,
        target: pd.Series,
        categorical_features: Sequence[str],
        feature_order: Sequence[str],
        splits: Iterable[tuple[np.ndarray, np.ndarray]],
        fold_labels: Sequence[str] | None,
        use_price_encoding: bool,
        smoke: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        prediction_path, fold_path = self._checkpoint_paths(key)
        if prediction_path.exists() and fold_path.exists() and not self.config.force:
            saved = np.load(prediction_path)
            prediction = saved["prediction"]
            fold = saved["fold"].astype(int)
            if len(prediction) == len(features):
                return prediction, fold, json.loads(fold_path.read_text())

        prediction = np.full(len(features), np.nan, dtype=float)
        fold_assignment = np.full(len(features), -1, dtype=int)
        test_counts = np.zeros(len(features), dtype=int)
        fold_records: list[dict] = []
        materialized = list(splits)
        for fold_number, (train_rows, test_rows) in enumerate(materialized, start=1):
            label = (
                fold_labels[fold_number - 1] if fold_labels else f"fold_{fold_number}"
            )
            print(
                f"[{protocol} rep={repetition}] fold {fold_number}/{len(materialized)} "
                f"train={len(train_rows):,} test={len(test_rows):,} ({label})",
                flush=True,
            )
            fold_prediction = fit_predict(
                features.iloc[train_rows],
                target.iloc[train_rows],
                features.iloc[test_rows],
                categorical_features,
                feature_order,
                use_price_encoding=use_price_encoding,
                smoke=smoke,
            )
            prediction[test_rows] = fold_prediction
            fold_assignment[test_rows] = fold_number
            test_counts[test_rows] += 1
            record = {
                "protocol": protocol,
                "repetition": int(repetition),
                "fold": int(fold_number),
                "fold_label": str(label),
                "train_n": int(len(train_rows)),
                "test_n": int(len(test_rows)),
                "use_price_encoding": bool(use_price_encoding),
            }
            record.update(
                metric_record(target.iloc[test_rows].to_numpy(), fold_prediction)
            )
            fold_records.append(record)

        if (
            np.isnan(prediction).any()
            or (fold_assignment < 0).any()
            or not np.all(test_counts == 1)
        ):
            raise AssertionError(f"{protocol} did not predict every row exactly once")
        np.savez_compressed(
            prediction_path, prediction=prediction, fold=fold_assignment
        )
        _json_dump(fold_path, fold_records)
        return prediction, fold_assignment, fold_records

    def run_smoke(self) -> None:
        rng = np.random.default_rng(42)
        selected = np.sort(
            rng.choice(
                np.arange(len(self.data.features)),
                size=min(1_000, len(self.data.features)),
                replace=False,
            )
        )
        smoke_data = self.data.subset(selected)
        splits = random_splits(len(smoke_data.features), n_splits=2, seed=42)
        prediction, _, records = self._run_oof(
            key="smoke__random_2fold",
            protocol="smoke_random",
            repetition=42,
            features=smoke_data.features,
            target=smoke_data.target,
            categorical_features=smoke_data.categorical_features,
            feature_order=smoke_data.feature_order,
            splits=splits,
            fold_labels=None,
            use_price_encoding=True,
            smoke=True,
        )
        result = metric_record(smoke_data.target.to_numpy(), prediction)
        result["folds"] = records
        _json_dump(self.output_dir / "smoke_test.json", result)
        print(f"Smoke test complete: MAE ${result['mae']:,.0f}", flush=True)

    def run_main(self) -> None:
        frames: list[pd.DataFrame] = []
        fold_rows: list[dict] = []
        data = self.data
        for seed in self.config.seeds:
            for kind, protocol in (
                ("random", "random_reference"),
                ("make_model", "holdout_make_model"),
                ("make", "holdout_make"),
            ):
                splits, labels = standard_splits(
                    kind,
                    data.metadata,
                    n_splits=self.config.n_splits,
                    seed=seed,
                )
                prediction, fold, records = self._run_oof(
                    key=f"main__{protocol}__seed_{seed}__enc_1",
                    protocol=protocol,
                    repetition=seed,
                    features=data.features,
                    target=data.target,
                    categorical_features=data.categorical_features,
                    feature_order=data.feature_order,
                    splits=splits,
                    fold_labels=labels,
                    use_price_encoding=True,
                )
                frames.append(
                    _long_frame(protocol, seed, prediction, fold, data, "primary")
                )
                fold_rows.extend(records)

        for spec, protocol in (
            (PRIMARY_MILEAGE, "holdout_mileage_primary"),
            (PRIMARY_AGE, "holdout_age_primary"),
            (SENSITIVITY_MILEAGE, "holdout_mileage_sensitivity"),
            (SENSITIVITY_AGE, "holdout_age_sensitivity"),
        ):
            splits, labels = band_splits(data.metadata, spec)
            prediction, fold, records = self._run_oof(
                key=f"main__{protocol}__seed_42__enc_1",
                protocol=protocol,
                repetition=42,
                features=data.features,
                target=data.target,
                categorical_features=data.categorical_features,
                feature_order=data.feature_order,
                splits=splits,
                fold_labels=labels,
                use_price_encoding=True,
            )
            frames.append(_long_frame(protocol, 42, prediction, fold, data, "primary"))
            fold_rows.extend(records)

        predictions = pd.concat(frames, ignore_index=True)
        summary, segments = self._summarize_main(predictions, data.metadata)
        pairwise = self._pairwise_protocol_summary(predictions, data.metadata)
        rollup = _segment_rollup(segments)
        predictions.to_csv(
            self.output_dir / "main_oof_predictions.csv.gz",
            index=False,
            compression="gzip",
        )
        pd.DataFrame(fold_rows).to_csv(
            self.output_dir / "main_fold_metrics.csv", index=False
        )
        summary.to_csv(self.output_dir / "main_summary.csv", index=False)
        segments.to_csv(self.output_dir / "segment_summary.csv", index=False)
        pairwise.to_csv(self.output_dir / "protocol_pairwise_summary.csv", index=False)
        rollup.to_csv(self.output_dir / "segment_rollup_min20.csv", index=False)
        _json_dump(
            self.output_dir / "main_summary.json", summary.to_dict(orient="records")
        )

    def _summarize_main(
        self, predictions: pd.DataFrame, metadata: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        per_row = predictions.groupby(["protocol", "row_id"], as_index=False).agg(
            actual=("actual", "first"),
            prediction=("prediction", "mean"),
            abs_error=("abs_error", "mean"),
            ape=("ape", "mean"),
            sq_error=("sq_error", "mean"),
        )
        reference = per_row.loc[
            per_row["protocol"] == "random_reference",
            ["row_id", "abs_error", "ape", "sq_error"],
        ].rename(
            columns={
                "abs_error": "reference_abs_error",
                "ape": "reference_ape",
                "sq_error": "reference_sq_error",
            }
        )
        if len(reference) != len(metadata):
            raise AssertionError("random reference does not cover every row")

        rows: list[dict] = []
        for protocol, group in per_row.groupby("protocol", sort=False):
            actual = group["actual"].to_numpy(float)
            absolute_error = group["abs_error"].to_numpy(float)
            squared_error = group["sq_error"].to_numpy(float)
            row = {
                "protocol": protocol,
                "n": int(len(group)),
                "mae": float(absolute_error.mean()),
                "rmse": float(np.sqrt(squared_error.mean())),
                "r2": float(
                    1 - squared_error.sum() / np.sum((actual - actual.mean()) ** 2)
                ),
                "mape_percent": float(100 * group["ape"].mean()),
                "mdape_percent": float(100 * group["ape"].median()),
                "wape_percent": float(100 * absolute_error.sum() / actual.sum()),
            }
            joined = group.merge(reference, on="row_id", how="left")
            reference_mae = float(joined["reference_abs_error"].mean())
            row["reference_mae"] = reference_mae
            row["mae_delta"] = float(row["mae"] - reference_mae)
            row["mae_ratio"] = float(row["mae"] / reference_mae)
            if protocol == "random_reference":
                row.update(
                    {
                        "delta_ci_low": 0.0,
                        "delta_ci_high": 0.0,
                        "ratio_ci_low": 1.0,
                        "ratio_ci_high": 1.0,
                        "bootstrap_reps": int(self.config.bootstrap_reps),
                        "n_clusters": int(metadata["make_model"].nunique()),
                    }
                )
            else:
                cluster_column = "make" if protocol == "holdout_make" else "make_model"
                ordered = joined.sort_values("row_id")
                cluster_values = metadata.sort_values("row_id")[
                    cluster_column
                ].to_numpy()
                row.update(
                    paired_cluster_bootstrap(
                        ordered["reference_abs_error"].to_numpy(),
                        ordered["abs_error"].to_numpy(),
                        cluster_values,
                        seed=42 + len(rows),
                        reps=self.config.bootstrap_reps,
                    )
                )
            rows.append(row)
        summary = pd.DataFrame(rows)

        segment_rows: list[dict] = []
        mappings = {
            "holdout_make_model": "make_model",
            "holdout_make": "make",
            "holdout_mileage_primary": "mileage_primary",
            "holdout_age_primary": "age_primary",
            "holdout_mileage_sensitivity": "mileage_sensitivity",
            "holdout_age_sensitivity": "age_sensitivity",
        }
        indexed_metadata = metadata.set_index("row_id")
        for protocol, segment_column in mappings.items():
            shifted = per_row.loc[per_row["protocol"] == protocol].copy()
            if shifted.empty:
                continue
            shifted[segment_column] = shifted["row_id"].map(
                indexed_metadata[segment_column]
            )
            joined = shifted.merge(reference, on="row_id", how="left")
            for segment, group in joined.groupby(segment_column, sort=False):
                reference_mae = float(group["reference_abs_error"].mean())
                shifted_mae = float(group["abs_error"].mean())
                segment_rows.append(
                    {
                        "protocol": protocol,
                        "segment_variable": segment_column,
                        "segment": str(segment),
                        "n": int(len(group)),
                        "median_actual_price": float(group["actual"].median()),
                        "reference_mae": reference_mae,
                        "shifted_mae": shifted_mae,
                        "mae_delta": shifted_mae - reference_mae,
                        "mae_ratio": shifted_mae / reference_mae,
                        "reference_mape_percent": float(
                            100 * group["reference_ape"].mean()
                        ),
                        "shifted_mape_percent": float(100 * group["ape"].mean()),
                    }
                )
        return summary, pd.DataFrame(segment_rows)

    def _pairwise_protocol_summary(
        self, predictions: pd.DataFrame, metadata: pd.DataFrame
    ) -> pd.DataFrame:
        per_row = predictions.groupby(["protocol", "row_id"], as_index=False).agg(
            abs_error=("abs_error", "mean")
        )
        wide = per_row.pivot(
            index="row_id", columns="protocol", values="abs_error"
        ).sort_index()
        protocols = list(wide.columns)
        rows: list[dict] = []
        ordered_metadata = metadata.set_index("row_id").loc[wide.index]
        for position, reference_protocol in enumerate(protocols):
            for shifted_protocol in protocols[position + 1 :]:
                reference = wide[reference_protocol].to_numpy(float)
                shifted = wide[shifted_protocol].to_numpy(float)
                cluster_column = (
                    "make"
                    if "holdout_make" in {reference_protocol, shifted_protocol}
                    else "make_model"
                )
                bootstrap = paired_cluster_bootstrap(
                    reference,
                    shifted,
                    ordered_metadata[cluster_column].to_numpy(),
                    seed=3_100 + len(rows),
                    reps=self.config.bootstrap_reps,
                )
                reference_mae = float(reference.mean())
                shifted_mae = float(shifted.mean())
                rows.append(
                    {
                        "reference_protocol": reference_protocol,
                        "shifted_protocol": shifted_protocol,
                        "n": int(len(reference)),
                        "reference_mae": reference_mae,
                        "shifted_mae": shifted_mae,
                        "mae_delta": shifted_mae - reference_mae,
                        "mae_ratio": shifted_mae / reference_mae,
                        "fraction_rows_higher_abs_error": float(
                            np.mean(shifted > reference)
                        ),
                        **bootstrap,
                    }
                )
        return pd.DataFrame(rows)

    def run_paired_ranges(self) -> None:
        data = self.data
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        prediction_frames: list[pd.DataFrame] = []
        fold_rows: list[dict] = []
        all_rows = np.arange(len(data.metadata))

        for spec_name, band_label, region_type in PAIRED_RANGES:
            spec = BAND_SPECS[spec_name]
            inside = all_rows[data.metadata[spec.name].to_numpy() == band_label]
            outside = all_rows[data.metadata[spec.name].to_numpy() != band_label]
            if len(inside) < self.config.n_splits:
                raise ValueError(
                    f"paired range has too few rows: {spec.name} / {band_label}"
                )

            absent_key = f"paired_v2__{spec.name}__{_slug(band_label)}__absent"
            absent_path, absent_record_path = self._checkpoint_paths(absent_key)
            if (
                absent_path.exists()
                and absent_record_path.exists()
                and not self.config.force
            ):
                saved = np.load(absent_path)
                absent_prediction = saved["prediction"]
                absent_record = json.loads(absent_record_path.read_text())
                if len(absent_prediction) != len(inside):
                    absent_prediction = np.array([])
            else:
                absent_prediction = np.array([])

            if len(absent_prediction) == 0:
                print(
                    f"[paired {spec.name} {band_label}] fitting one range-absent model",
                    flush=True,
                )
                absent_prediction = fit_predict(
                    data.features.iloc[outside],
                    data.target.iloc[outside],
                    data.features.iloc[inside],
                    data.categorical_features,
                    data.feature_order,
                    use_price_encoding=True,
                )
                absent_record = {
                    "feature": spec.source_column,
                    "band_definition": spec.name,
                    "band": band_label,
                    "region_type": region_type,
                    "condition": "range_absent",
                    "repetition": 0,
                    "fold": 0,
                    "train_n": int(len(outside)),
                    "test_n": int(len(inside)),
                    "in_range_training_rows": 0,
                }
                absent_record.update(
                    metric_record(
                        data.target.iloc[inside].to_numpy(), absent_prediction
                    )
                )
                np.savez_compressed(
                    absent_path, prediction=absent_prediction, row_ids=inside
                )
                _json_dump(absent_record_path, absent_record)
            fold_rows.append(absent_record)

            inside_fingerprints = (
                data.metadata.iloc[inside]["content_fingerprint"].astype(str).to_numpy()
            )
            for repetition in self.config.seeds:
                represented_key = (
                    f"paired_v2__{spec.name}__{_slug(band_label)}"
                    f"__represented__seed_{repetition}"
                )
                prediction_path, fold_path = self._checkpoint_paths(represented_key)
                if (
                    prediction_path.exists()
                    and fold_path.exists()
                    and not self.config.force
                ):
                    saved = np.load(prediction_path)
                    represented_prediction = saved["prediction"]
                    fold_assignment = saved["fold"].astype(int)
                    records = json.loads(fold_path.read_text())
                    if len(represented_prediction) != len(inside):
                        represented_prediction = np.array([])
                else:
                    represented_prediction = np.array([])

                if len(represented_prediction) == 0:
                    represented_prediction = np.full(len(inside), np.nan)
                    fold_assignment = np.full(len(inside), -1, dtype=int)
                    test_counts = np.zeros(len(inside), dtype=int)
                    records: list[dict] = []
                    inner_splits = balanced_group_splits(
                        inside_fingerprints,
                        n_splits=self.config.n_splits,
                        seed=repetition,
                    )
                    for fold_number, (
                        inside_train_positions,
                        inside_test_positions,
                    ) in enumerate(inner_splits, start=1):
                        test_rows = inside[inside_test_positions]
                        represented_inside_rows = inside[inside_train_positions]
                        remaining_outside, match_diagnostics = matched_outside_removals(
                            represented_inside_rows,
                            outside,
                            data.metadata,
                            seed=(
                                100_000
                                + 1_000 * repetition
                                + 10 * fold_number
                                + list(BAND_SPECS).index(spec.name)
                            ),
                        )
                        train_rows = np.concatenate(
                            [represented_inside_rows, remaining_outside]
                        )
                        if len(train_rows) != len(outside):
                            raise AssertionError("matched-size training counts differ")
                        if np.intersect1d(train_rows, test_rows).size:
                            raise AssertionError(
                                "paired training and test rows overlap"
                            )
                        train_fingerprints = set(
                            data.metadata.iloc[train_rows][
                                "content_fingerprint"
                            ].astype(str)
                        )
                        test_fingerprints = set(
                            data.metadata.iloc[test_rows]["content_fingerprint"].astype(
                                str
                            )
                        )
                        if train_fingerprints.intersection(test_fingerprints):
                            raise AssertionError(
                                "exact-content duplicate crosses a paired fold"
                            )

                        print(
                            f"[paired {spec.name} {band_label} rep={repetition}] "
                            f"fold {fold_number}/{self.config.n_splits} "
                            f"train={len(train_rows):,} test={len(test_rows):,}",
                            flush=True,
                        )
                        fold_prediction = fit_predict(
                            data.features.iloc[train_rows],
                            data.target.iloc[train_rows],
                            data.features.iloc[test_rows],
                            data.categorical_features,
                            data.feature_order,
                            use_price_encoding=True,
                        )
                        represented_prediction[inside_test_positions] = fold_prediction
                        fold_assignment[inside_test_positions] = fold_number
                        test_counts[inside_test_positions] += 1
                        record = {
                            "feature": spec.source_column,
                            "band_definition": spec.name,
                            "band": band_label,
                            "region_type": region_type,
                            "condition": "range_represented",
                            "repetition": int(repetition),
                            "fold": int(fold_number),
                            "train_n": int(len(train_rows)),
                            "test_n": int(len(test_rows)),
                            "in_range_training_rows": int(len(represented_inside_rows)),
                            **match_diagnostics,
                        }
                        record.update(
                            metric_record(
                                data.target.iloc[test_rows].to_numpy(), fold_prediction
                            )
                        )
                        records.append(record)
                    if np.isnan(represented_prediction).any() or not np.all(
                        test_counts == 1
                    ):
                        raise AssertionError(
                            "paired represented condition did not score each row once"
                        )
                    np.savez_compressed(
                        prediction_path,
                        prediction=represented_prediction,
                        row_ids=inside,
                        fold=fold_assignment,
                    )
                    _json_dump(fold_path, records)
                fold_rows.extend(records)

                for condition, prediction in (
                    ("range_represented", represented_prediction),
                    ("range_absent", absent_prediction),
                ):
                    actual = data.target.iloc[inside].to_numpy(float)
                    frame = data.metadata.iloc[inside][
                        ["row_id", "make", "make_model", "mileage", "car_age"]
                    ].copy()
                    frame["feature"] = spec.source_column
                    frame["band_definition"] = spec.name
                    frame["band"] = band_label
                    frame["region_type"] = region_type
                    frame["condition"] = condition
                    frame["repetition"] = int(repetition)
                    frame["fold"] = fold_assignment
                    frame["actual"] = actual
                    frame["prediction"] = prediction
                    frame["error"] = prediction - actual
                    frame["abs_error"] = np.abs(frame["error"])
                    frame["ape"] = frame["abs_error"] / actual
                    prediction_frames.append(frame)

        predictions = pd.concat(prediction_frames, ignore_index=True)
        summary_rows: list[dict] = []
        keys = ["feature", "band_definition", "band", "region_type"]
        for key_values, group in predictions.groupby(keys, sort=False):
            by_row = group.groupby(["condition", "row_id"], as_index=False).agg(
                actual=("actual", "first"),
                make_model=("make_model", "first"),
                abs_error=("abs_error", "mean"),
                ape=("ape", "mean"),
            )
            represented = by_row[
                by_row["condition"] == "range_represented"
            ].sort_values("row_id")
            absent = by_row[by_row["condition"] == "range_absent"].sort_values("row_id")
            if not np.array_equal(
                represented["row_id"].to_numpy(), absent["row_id"].to_numpy()
            ):
                raise AssertionError("paired test rows are not identical")
            represented_mae = float(represented["abs_error"].mean())
            absent_mae = float(absent["abs_error"].mean())
            bootstrap = paired_cluster_bootstrap(
                represented["abs_error"].to_numpy(),
                absent["abs_error"].to_numpy(),
                absent["make_model"].to_numpy(),
                seed=77 + len(summary_rows),
                reps=self.config.bootstrap_reps,
            )
            row = {
                **dict(zip(keys, key_values)),
                "n": int(len(absent)),
                "median_actual_price": float(absent["actual"].median()),
                "represented_mae": represented_mae,
                "absent_mae": absent_mae,
                "mae_delta": absent_mae - represented_mae,
                "mae_ratio": absent_mae / represented_mae,
                "represented_mape_percent": float(100 * represented["ape"].mean()),
                "absent_mape_percent": float(100 * absent["ape"].mean()),
                "represented_mae_across_repetition_sd": float(
                    group[group["condition"] == "range_represented"]
                    .groupby("repetition")["abs_error"]
                    .mean()
                    .std(ddof=0)
                ),
                **bootstrap,
            }
            summary_rows.append(row)

        summary = pd.DataFrame(summary_rows)
        predictions.to_csv(
            self.output_dir / "paired_range_predictions.csv.gz",
            index=False,
            compression="gzip",
        )
        pd.DataFrame(fold_rows).to_csv(
            self.output_dir / "paired_range_fold_metrics.csv", index=False
        )
        summary.to_csv(self.output_dir / "paired_range_summary.csv", index=False)
        _json_dump(
            self.output_dir / "paired_range_summary.json",
            summary.to_dict(orient="records"),
        )

    def _run_three_protocols(
        self,
        data: StudyData,
        *,
        analysis_set: str,
        use_price_encoding: bool,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        frames: list[pd.DataFrame] = []
        fold_rows: list[dict] = []
        for kind, protocol in (
            ("random", "random_reference"),
            ("make_model", "holdout_make_model"),
            ("make", "holdout_make"),
        ):
            splits, labels = standard_splits(
                kind,
                data.metadata,
                n_splits=self.config.n_splits,
                seed=42,
            )
            prediction, fold, records = self._run_oof(
                key=(
                    f"sensitivity__{analysis_set}__{protocol}"
                    f"__enc_{int(use_price_encoding)}"
                ),
                protocol=protocol,
                repetition=42,
                features=data.features,
                target=data.target,
                categorical_features=data.categorical_features,
                feature_order=data.feature_order,
                splits=splits,
                fold_labels=labels,
                use_price_encoding=use_price_encoding,
            )
            frames.append(
                _long_frame(
                    protocol, 42, prediction, fold, data, analysis_set=analysis_set
                )
            )
            fold_rows.extend(records)
        return pd.concat(frames, ignore_index=True), pd.DataFrame(fold_rows)

    def run_sensitivities(self) -> None:
        data = self.data
        no_encoding, no_encoding_folds = self._run_three_protocols(
            data,
            analysis_set="no_price_encoding",
            use_price_encoding=False,
        )

        keep = ~data.metadata["content_fingerprint"].duplicated(keep="first")
        deduplicated_data = data.subset(np.flatnonzero(keep.to_numpy()))
        deduplicated, deduplicated_folds = self._run_three_protocols(
            deduplicated_data,
            analysis_set="strict_exact_content_dedup",
            use_price_encoding=True,
        )

        proxy_drop = {"mileage_bucket", "miles_per_year"}
        proxy_features = [
            column for column in data.feature_order if column not in proxy_drop
        ]
        proxy_categories = [
            column for column in data.categorical_features if column not in proxy_drop
        ]
        proxy_frames: list[pd.DataFrame] = []
        proxy_fold_rows: list[dict] = []
        for kind, protocol in (
            ("random", "random_reference"),
            ("mileage", "holdout_mileage_primary"),
        ):
            if kind == "random":
                splits, labels = standard_splits(
                    "random", data.metadata, n_splits=self.config.n_splits, seed=42
                )
            else:
                splits, labels = band_splits(data.metadata, PRIMARY_MILEAGE)
            prediction, fold, records = self._run_oof(
                key=f"sensitivity__mileage_proxy_reduced__{protocol}__enc_1",
                protocol=protocol,
                repetition=42,
                features=data.features,
                target=data.target,
                categorical_features=proxy_categories,
                feature_order=proxy_features,
                splits=splits,
                fold_labels=labels,
                use_price_encoding=True,
            )
            proxy_frames.append(
                _long_frame(
                    protocol,
                    42,
                    prediction,
                    fold,
                    data,
                    analysis_set="mileage_proxy_reduced",
                )
            )
            proxy_fold_rows.extend(records)
        proxy_reduced = pd.concat(proxy_frames, ignore_index=True)

        manufacturer_features = data.features.copy()
        manufacturer_features["make"] = pd.Categorical(
            data.metadata["make"].astype(str)
        )
        manufacturer_feature_order = [
            column for column in data.feature_order if column != ENCODING_FEATURE
        ] + ["make", ENCODING_FEATURE]
        manufacturer_categories = list(
            dict.fromkeys([*data.categorical_features, "make"])
        )
        manufacturer_data = StudyData(
            features=manufacturer_features,
            target=data.target,
            metadata=data.metadata,
            categorical_features=manufacturer_categories,
            feature_order=manufacturer_feature_order,
            source_path=data.source_path,
        )
        manufacturer_aware, manufacturer_folds = self._run_three_protocols(
            manufacturer_data,
            analysis_set="manufacturer_aware_model",
            use_price_encoding=True,
        )

        predictions = pd.concat(
            [no_encoding, deduplicated, proxy_reduced, manufacturer_aware],
            ignore_index=True,
        )
        summary_rows: list[dict] = []
        for analysis_set, analysis in predictions.groupby("analysis_set", sort=False):
            per_row = analysis.groupby(["protocol", "row_id"], as_index=False).agg(
                actual=("actual", "first"),
                abs_error=("abs_error", "mean"),
                ape=("ape", "mean"),
                sq_error=("sq_error", "mean"),
            )
            reference = per_row[per_row["protocol"] == "random_reference"].set_index(
                "row_id"
            )["abs_error"]
            for protocol, group in per_row.groupby("protocol", sort=False):
                mae = float(group["abs_error"].mean())
                reference_mae = float(group["row_id"].map(reference).mean())
                summary_rows.append(
                    {
                        "analysis_set": analysis_set,
                        "protocol": protocol,
                        "n": int(len(group)),
                        "mae": mae,
                        "mape_percent": float(100 * group["ape"].mean()),
                        "reference_mae": reference_mae,
                        "mae_delta": mae - reference_mae,
                        "mae_ratio": mae / reference_mae,
                    }
                )

        predictions.to_csv(
            self.output_dir / "sensitivity_predictions.csv.gz",
            index=False,
            compression="gzip",
        )
        pd.concat(
            [
                no_encoding_folds.assign(analysis_set="no_price_encoding"),
                deduplicated_folds.assign(analysis_set="strict_exact_content_dedup"),
                pd.DataFrame(proxy_fold_rows).assign(
                    analysis_set="mileage_proxy_reduced"
                ),
                manufacturer_folds.assign(analysis_set="manufacturer_aware_model"),
            ],
            ignore_index=True,
        ).to_csv(self.output_dir / "sensitivity_fold_metrics.csv", index=False)
        pd.DataFrame(summary_rows).to_csv(
            self.output_dir / "sensitivity_summary.csv", index=False
        )
        _json_dump(self.output_dir / "sensitivity_summary.json", summary_rows)


def _long_frame(
    protocol: str,
    repetition: int,
    prediction: np.ndarray,
    fold: np.ndarray,
    data: StudyData,
    analysis_set: str,
) -> pd.DataFrame:
    actual = data.target.to_numpy(dtype=float)
    result = data.metadata[
        [
            "row_id",
            "make",
            "make_model",
            "mileage",
            "car_age",
            "mileage_primary",
            "age_primary",
        ]
    ].copy()
    result["analysis_set"] = analysis_set
    result["protocol"] = protocol
    result["repetition"] = repetition
    result["fold"] = fold
    result["actual"] = actual
    result["prediction"] = prediction
    result["error"] = prediction - actual
    result["abs_error"] = np.abs(result["error"])
    result["ape"] = result["abs_error"] / result["actual"]
    result["sq_error"] = result["error"] ** 2
    return result


def _segment_rollup(segments: pd.DataFrame, minimum_n: int = 20) -> pd.DataFrame:
    eligible = segments[segments["n"] >= minimum_n].copy()
    rows: list[dict] = []
    for (protocol, variable), group in eligible.groupby(
        ["protocol", "segment_variable"], sort=False
    ):
        weights = group["n"].to_numpy(float)
        rows.append(
            {
                "protocol": protocol,
                "segment_variable": variable,
                "minimum_group_n": int(minimum_n),
                "eligible_groups": int(len(group)),
                "eligible_rows": int(group["n"].sum()),
                "listing_weighted_reference_mae": float(
                    np.average(group["reference_mae"], weights=weights)
                ),
                "listing_weighted_shifted_mae": float(
                    np.average(group["shifted_mae"], weights=weights)
                ),
                "equal_group_reference_mae": float(group["reference_mae"].mean()),
                "equal_group_shifted_mae": float(group["shifted_mae"].mean()),
                "equal_group_mean_mae_delta": float(group["mae_delta"].mean()),
                "equal_group_median_mae_ratio": float(group["mae_ratio"].median()),
            }
        )
    return pd.DataFrame(rows)


def _study_digest(data: StudyData, *, reference_year: int) -> str:
    digest = hashlib.sha256()
    digest.update(b"coverage-shift-checkpoint-v2\n")
    digest.update(f"code_version={EXPERIMENT_CODE_VERSION}\n".encode("utf-8"))
    digest.update(f"code_sha256={_package_source_digest()}\n".encode("utf-8"))
    digest.update(
        json.dumps(_runtime_versions(), sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    digest.update(b"\n")
    digest.update(f"reference_year={reference_year}\n".encode("utf-8"))
    model_spec = {
        "parameters": MODEL_PARAMETERS,
        "monotone_directions": MONOTONE_DIRECTIONS,
        "encoding_smoothing": ENCODING_SMOOTHING,
    }
    digest.update(
        json.dumps(model_spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    digest.update(b"\n")
    digest.update("\0".join(data.feature_order).encode("utf-8"))
    digest.update(b"\n")
    digest.update("\0".join(data.categorical_features).encode("utf-8"))
    digest.update(b"\n")
    for value in data.metadata["content_fingerprint"].astype(str):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _package_source_digest() -> str:
    """Fingerprint the experiment package so changed code cannot reuse checkpoints."""
    digest = hashlib.sha256()
    package_dir = Path(__file__).resolve().parent
    for source in sorted(package_dir.glob("*.py")):
        digest.update(source.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _runtime_versions() -> dict[str, str]:
    """Versions that can affect predictions and checkpoint compatibility."""
    return {
        "python": platform.python_version(),
        "numpy": importlib.metadata.version("numpy"),
        "pandas": importlib.metadata.version("pandas"),
        "scikit_learn": importlib.metadata.version("scikit-learn"),
        "lightgbm": importlib.metadata.version("lightgbm"),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(value: object) -> str:
    return (
        "".join(character if character.isalnum() else "_" for character in str(value))
        .strip("_")
        .lower()
    )


def _json_dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
