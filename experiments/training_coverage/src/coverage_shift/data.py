from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


TARGET_COLUMN = "price"
ENCODING_FEATURE = "make_model_price_enc"

MODEL_INPUT_FEATURES = (
    "make_model",
    "year",
    "mileage",
    "car_age",
    "miles_per_year",
    "mileage_bucket",
    "is_luxury_make",
    "is_new",
    "trim",
    "trim_tier",
    "title_status",
    "accident_history",
    "owner_count",
    "cpo",
    "drivetrain",
    "engine_size_l",
    "transmission_type",
    "fuel_type_group",
    "mpg",
    "car_color",
    "body_style",
)

CATEGORICAL_FEATURES = (
    "make_model",
    "trim",
    "body_style",
    "mileage_bucket",
    "drivetrain",
    "fuel_type_group",
    "transmission_type",
    "car_color",
    "title_status",
    "accident_history",
)

MILEAGE_BUCKET_EDGES = (-1, 15_000, 60_000, 120_000, np.inf)
MILEAGE_BUCKET_LABELS = ("very_low", "low", "medium", "high")


@dataclass(frozen=True)
class BandSpec:
    name: str
    source_column: str
    edges: tuple[float, ...]
    labels: tuple[str, ...]

    def apply(self, values: pd.Series) -> pd.Series:
        assigned = pd.cut(
            pd.to_numeric(values, errors="coerce"),
            bins=list(self.edges),
            labels=list(self.labels),
            include_lowest=True,
            right=False,
        )
        if assigned.isna().any():
            missing = int(assigned.isna().sum())
            raise ValueError(f"{self.name} left {missing} rows unassigned")
        return assigned.astype(str)


PRIMARY_MILEAGE = BandSpec(
    "mileage_primary",
    "mileage",
    (-np.inf, 15_000, 35_000, 60_000, 100_000, np.inf),
    ("0-14,999", "15,000-34,999", "35,000-59,999", "60,000-99,999", "100,000+"),
)
PRIMARY_AGE = BandSpec(
    "age_primary",
    "car_age",
    (-np.inf, 2, 4, 6, 9, np.inf),
    ("0-1", "2-3", "4-5", "6-8", "9+"),
)
SENSITIVITY_MILEAGE = BandSpec(
    "mileage_sensitivity",
    "mileage",
    (-np.inf, 10_000, 25_000, 50_000, 100_000, np.inf),
    ("0-9,999", "10,000-24,999", "25,000-49,999", "50,000-99,999", "100,000+"),
)
SENSITIVITY_AGE = BandSpec(
    "age_sensitivity",
    "car_age",
    (-np.inf, 1, 3, 5, 9, np.inf),
    ("0", "1-2", "3-4", "5-8", "9+"),
)

BAND_SPECS = {
    spec.name: spec
    for spec in (PRIMARY_MILEAGE, PRIMARY_AGE, SENSITIVITY_MILEAGE, SENSITIVITY_AGE)
}

PAIRED_RANGES = (
    (PRIMARY_MILEAGE.name, "35,000-59,999", "interior"),
    (PRIMARY_MILEAGE.name, "100,000+", "upper_tail"),
    (PRIMARY_AGE.name, "4-5", "interior"),
    (PRIMARY_AGE.name, "9+", "upper_tail"),
)


@dataclass
class StudyData:
    features: pd.DataFrame
    target: pd.Series
    metadata: pd.DataFrame
    categorical_features: list[str]
    feature_order: list[str]
    source_path: Path | None = None

    def subset(self, indices: Sequence[int]) -> "StudyData":
        idx = np.asarray(indices, dtype=int)
        metadata = self.metadata.iloc[idx].reset_index(drop=True).copy()
        metadata["row_id"] = np.arange(len(metadata), dtype=int)
        return StudyData(
            features=self.features.iloc[idx].reset_index(drop=True),
            target=self.target.iloc[idx].reset_index(drop=True),
            metadata=metadata,
            categorical_features=list(self.categorical_features),
            feature_order=list(self.feature_order),
            source_path=self.source_path,
        )


def load_study_data(path: str | Path, reference_year: int = 2026) -> StudyData:
    source = Path(path)
    frame = pd.read_csv(source)
    return prepare_study_data(frame, reference_year=reference_year, source_path=source)


def prepare_study_data(
    frame: pd.DataFrame,
    *,
    reference_year: int = 2026,
    source_path: Path | None = None,
) -> StudyData:
    """Prepare a model-ready listing table for the fixed experiment.

    The input is the processed listing table used by the study, not raw collection
    output. Age-derived fields are rebuilt from a fixed reference year so rerunning
    the experiment later does not change the design.
    """
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"required column missing: {TARGET_COLUMN}")

    source_cleaned = frame.copy().dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    numeric_target = pd.to_numeric(source_cleaned[TARGET_COLUMN], errors="coerce")
    valid = numeric_target.notna() & np.isfinite(numeric_target) & (numeric_target > 0)
    fingerprint_source = source_cleaned.loc[valid].reset_index(drop=True)
    cleaned = fingerprint_source.copy()
    cleaned[TARGET_COLUMN] = numeric_target.loc[valid].reset_index(drop=True)
    if cleaned.empty:
        raise ValueError("no rows have a positive numeric price")

    required = {"make", "year", "mileage"}
    if "make_model" not in cleaned.columns:
        required.add("model")
    missing = sorted(required.difference(cleaned.columns))
    if missing:
        raise ValueError(f"required columns missing: {', '.join(missing)}")

    cleaned["make"] = cleaned["make"].astype("string")
    if "make_model" not in cleaned.columns:
        cleaned["model"] = cleaned["model"].astype("string")
        cleaned["make_model"] = cleaned["make"] + "__" + cleaned["model"]

    cleaned["year"] = pd.to_numeric(cleaned["year"], errors="coerce")
    cleaned["mileage"] = pd.to_numeric(cleaned["mileage"], errors="coerce")
    invalid_year = cleaned["year"].isna() | ~np.isfinite(cleaned["year"])
    invalid_mileage = (
        cleaned["mileage"].isna()
        | ~np.isfinite(cleaned["mileage"])
        | (cleaned["mileage"] < 0)
    )
    if invalid_year.any() or invalid_mileage.any():
        raise ValueError(
            "year must be finite and mileage must be finite and nonnegative "
            "for every eligible row"
        )
    cleaned["car_age"] = (reference_year - cleaned["year"]).clip(lower=0)
    age_for_rate = cleaned["car_age"].replace(0, 1)
    cleaned["miles_per_year"] = (cleaned["mileage"] / age_for_rate).replace(
        [np.inf, -np.inf], np.nan
    )
    cleaned["mileage_bucket"] = pd.cut(
        cleaned["mileage"],
        bins=list(MILEAGE_BUCKET_EDGES),
        labels=list(MILEAGE_BUCKET_LABELS),
    ).astype("string")

    if cleaned[["make", "make_model", "year", "mileage", "car_age"]].isna().any().any():
        raise ValueError(
            "make, make_model, year, and mileage must be present for every eligible row"
        )

    target = cleaned[TARGET_COLUMN].astype(float).reset_index(drop=True)
    missing_inputs = [
        column for column in MODEL_INPUT_FEATURES if column not in cleaned.columns
    ]
    if missing_inputs:
        raise ValueError(
            "processed table is missing registered model inputs: "
            + ", ".join(missing_inputs)
        )
    features = cleaned[list(MODEL_INPUT_FEATURES)].copy()

    categorical = [
        column for column in CATEGORICAL_FEATURES if column in features.columns
    ]
    for column in categorical:
        features[column] = features[column].astype("category")
    for column in features.columns:
        if column not in categorical:
            features[column] = pd.to_numeric(features[column], errors="coerce")

    all_missing = [
        column for column in features.columns if features[column].isna().all()
    ]
    if all_missing:
        raise ValueError(
            "registered model inputs cannot be entirely missing: "
            + ", ".join(all_missing)
        )

    metadata_columns = [
        column
        for column in (
            "listing_id",
            "listing_key",
            "make",
            "model",
            "make_model",
            "mileage",
            "car_age",
            "year",
            "price",
        )
        if column in cleaned.columns
    ]
    metadata = cleaned[metadata_columns].copy().reset_index(drop=True)
    fingerprint_columns = [
        column
        for column in fingerprint_source.columns
        if column not in {"listing_id", "listing_key", "zip", "region"}
    ]
    metadata["content_fingerprint"] = pd.util.hash_pandas_object(
        fingerprint_source[fingerprint_columns].reset_index(drop=True), index=False
    ).astype(str)
    metadata["row_id"] = np.arange(len(metadata), dtype=int)
    for spec in BAND_SPECS.values():
        metadata[spec.name] = spec.apply(metadata[spec.source_column])

    if len(features) != len(target) or len(features) != len(metadata):
        raise AssertionError("feature, target, and metadata rows are not aligned")
    if not np.array_equal(
        features["make_model"].astype(str).to_numpy(),
        metadata["make_model"].astype(str).to_numpy(),
    ):
        raise AssertionError(
            "make_model metadata is not aligned with the feature frame"
        )

    return StudyData(
        features=features.reset_index(drop=True),
        target=target,
        metadata=metadata,
        categorical_features=categorical,
        feature_order=features.columns.tolist() + [ENCODING_FEATURE],
        source_path=source_path,
    )


def synthetic_listings(
    n_rows: int = 600, reference_year: int = 2026, seed: int = 42
) -> pd.DataFrame:
    """Create a deterministic model-ready table for a quick end-to-end check."""
    if n_rows < 50:
        raise ValueError("synthetic smoke data needs at least 50 rows")
    rng = np.random.default_rng(seed)
    makes = np.array(
        ["atlas", "brook", "cedar", "delta", "elm", "fjord", "grove", "harbor"]
    )
    model_names = np.array(["one", "two", "three", "four"])
    make_idx = rng.integers(0, len(makes), size=n_rows)
    model_idx = rng.integers(0, len(model_names), size=n_rows)
    make = makes[make_idx]
    model = model_names[model_idx]
    age = rng.integers(0, 17, size=n_rows)
    year = reference_year - age
    annual_miles = rng.lognormal(mean=np.log(8_500), sigma=0.5, size=n_rows)
    mileage = np.maximum(
        0, age * annual_miles + rng.normal(2_500, 4_000, size=n_rows)
    ).round()
    mileage = np.clip(mileage, 0, 220_000)
    luxury = (make_idx >= 6).astype(int)
    is_new = (age == 0).astype(int)
    trim_tier = rng.choice([0, 1, 2], p=[0.66, 0.25, 0.09], size=n_rows)
    make_effect = make_idx * 1_400 + luxury * 15_000
    model_effect = model_idx * 1_100
    price = (
        31_000
        + make_effect
        + model_effect
        + trim_tier * 7_000
        - age * 1_250
        - mileage * 0.055
        + rng.normal(0, 3_000, size=n_rows)
    )
    price = np.clip(price, 2_500, None).round(0)

    frame = pd.DataFrame(
        {
            "listing_id": np.arange(n_rows),
            "listing_key": [f"sample-{i}" for i in range(n_rows)],
            "zip": rng.choice([10_001, 30_303, 60_601, 90_001], size=n_rows),
            "region": rng.choice(["north", "south", "east", "west"], size=n_rows),
            "price": price,
            "make": make,
            "model": model,
            "make_model": np.char.add(np.char.add(make, "__"), model),
            "year": year,
            "mileage": mileage,
            "car_age": age,
            "miles_per_year": mileage / np.maximum(age, 1),
            "price_per_mile": price / np.maximum(mileage, 1),
            "is_luxury_make": luxury,
            "is_new": is_new,
            "trim": np.array(["base", "premium", "sport"])[trim_tier],
            "trim_tier": trim_tier,
            "title_status": rng.choice(["clean", "other"], p=[0.96, 0.04], size=n_rows),
            "accident_history": rng.choice(
                ["none", "reported"], p=[0.82, 0.18], size=n_rows
            ),
            "owner_count": np.where(is_new, 0, rng.integers(1, 5, size=n_rows)),
            "cpo": rng.integers(0, 2, size=n_rows),
            "drivetrain": rng.choice(["fwd", "rwd", "awd"], size=n_rows),
            "engine_size_l": rng.choice([1.5, 2.0, 2.5, 3.0, 4.0], size=n_rows),
            "transmission_type": rng.choice(
                ["automatic", "manual", "cvt"], size=n_rows
            ),
            "fuel_type_group": rng.choice(["gas", "hybrid", "electric"], size=n_rows),
            "mpg": rng.normal(28, 6, size=n_rows).clip(12, 55),
            "car_color": rng.choice(["black", "white", "silver", "blue"], size=n_rows),
            "body_style": rng.choice(["sedan", "suv", "truck", "coupe"], size=n_rows),
        }
    )
    frame["mileage_bucket"] = pd.cut(
        frame["mileage"],
        bins=list(MILEAGE_BUCKET_EDGES),
        labels=list(MILEAGE_BUCKET_LABELS),
    ).astype("string")
    return frame
