"""Regression tests for the shared training and serving feature contract."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import predict_listing as prediction  # noqa: E402
from train_price_model import MODEL_FEATURES, prepare_features  # noqa: E402


def test_prepare_features_ignores_unexpected_columns():
    row = {
        "price": 20_000,
        "make_model": "honda__civic",
        "year": 2020,
        "mileage": 40_000,
        "car_age": 6,
        "miles_per_year": 6_667,
        "mileage_bucket": "low",
        "is_luxury_make": 0,
        "is_new": 0,
        "trim": "sport",
        "trim_tier": 0,
        "title_status": "clean",
        "accident_history": "no_accident",
        "owner_count": 1,
        "cpo": 0,
        "drivetrain": "fwd",
        "engine_size_l": 2.0,
        "transmission_type": "automatic",
        "fuel_type_group": "gas",
        "mpg": 32,
        "car_color": "black",
        "body_style": "sedan",
        "target_clone": 20_000,
    }

    x, y, names, _ = prepare_features(pd.DataFrame([row]))

    assert names == MODEL_FEATURES
    assert "target_clone" not in x.columns
    assert y.tolist() == [20_000]


def test_reference_year_is_shared_by_single_listing_features():
    listing = prediction.assemble_listing(
        "Honda", "Civic", 2020, 40_000, reference_year=2030,
    )

    assert listing["car_age"] == 10
    assert listing["miles_per_year"] == 4_000


def test_new_vehicle_inputs_match_training_status_rules():
    listing = prediction.assemble_listing(
        "Honda", "Civic", 2026, 0, owner_count=None, cpo=True, is_new=True,
    )

    assert listing["is_new"] == 1
    assert listing["owner_count"] == 0
    assert listing["cpo"] == 0


def test_grouped_categories_use_fitted_unknown_and_rare_buckets():
    levels = {
        "trim": ["sport", "__unknown__", "__rare__"],
        "body_style": ["sedan", "__unknown__", "__rare__"],
        "car_color": ["black", "__unknown__", "__rare__"],
    }
    row = prediction._row_from_listing(
        {"trim": None, "body_style": "spaceship", "car_color": "black"},
        ["trim", "body_style", "car_color"],
        ["trim", "body_style", "car_color"],
        levels,
    )

    assert str(row.at[0, "trim"]) == "__unknown__"
    assert str(row.at[0, "body_style"]) == "__rare__"
    assert str(row.at[0, "car_color"]) == "black"
