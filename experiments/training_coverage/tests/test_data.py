import numpy as np
import pandas as pd
import pytest

from coverage_shift.data import (
    ENCODING_FEATURE,
    MODEL_INPUT_FEATURES,
    PRIMARY_AGE,
    PRIMARY_MILEAGE,
    prepare_study_data,
    synthetic_listings,
)


def test_preparation_rebuilds_age_and_excludes_identifiers_and_leakage():
    frame = synthetic_listings(n_rows=80, reference_year=2026)
    frame.loc[0, "car_age"] = 999
    frame.loc[0, "miles_per_year"] = -1
    frame.loc[1, "price"] = 0
    frame.loc[2, "price"] = np.nan
    frame["target_clone"] = frame["price"]

    data = prepare_study_data(frame, reference_year=2026)

    assert len(data.features) == 78
    assert data.metadata.loc[0, "car_age"] == 2026 - frame.loc[0, "year"]
    assert data.features.loc[0, "miles_per_year"] >= 0
    assert data.target.gt(0).all()
    assert ENCODING_FEATURE not in data.features
    assert "target_clone" not in data.features
    assert ENCODING_FEATURE == data.feature_order[-1]
    assert data.features.columns.tolist() == list(MODEL_INPUT_FEATURES)
    assert len(data.feature_order) == 22
    assert {"listing_id", "listing_key", "price_per_mile", "make", "model"}.isdisjoint(
        data.features.columns
    )


def test_content_fingerprint_ignores_identifiers_and_location():
    frame = synthetic_listings(n_rows=60)
    duplicate = frame.iloc[[0]].copy()
    duplicate["listing_id"] = 99_999
    duplicate["listing_key"] = "another-listing"
    duplicate["zip"] = 12_345
    duplicate["region"] = "elsewhere"
    frame = pd.concat([frame, duplicate], ignore_index=True)

    data = prepare_study_data(frame)

    assert (
        data.metadata.loc[0, "content_fingerprint"]
        == data.metadata.loc[60, "content_fingerprint"]
    )


def test_preparation_rejects_an_incomplete_registered_schema():
    frame = synthetic_listings(n_rows=60).drop(columns=["mpg"])

    with pytest.raises(ValueError, match="missing registered model inputs: mpg"):
        prepare_study_data(frame)


def test_preparation_rejects_invalid_mileage():
    frame = synthetic_listings(n_rows=60)
    frame.loc[0, "mileage"] = -1

    with pytest.raises(ValueError, match="mileage must be finite and nonnegative"):
        prepare_study_data(frame)


def test_registered_band_boundaries_are_left_closed():
    mileage = pd.Series(
        [0, 14_999, 15_000, 34_999, 35_000, 59_999, 60_000, 99_999, 100_000]
    )
    age = pd.Series([0, 1, 2, 3, 4, 5, 6, 8, 9])

    assert PRIMARY_MILEAGE.apply(mileage).tolist() == [
        "0-14,999",
        "0-14,999",
        "15,000-34,999",
        "15,000-34,999",
        "35,000-59,999",
        "35,000-59,999",
        "60,000-99,999",
        "60,000-99,999",
        "100,000+",
    ]
    assert PRIMARY_AGE.apply(age).tolist() == [
        "0-1",
        "0-1",
        "2-3",
        "2-3",
        "4-5",
        "4-5",
        "6-8",
        "6-8",
        "9+",
    ]
