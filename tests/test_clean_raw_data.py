"""Regression tests for raw-data cleaning edge cases."""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from clean_raw_data import clean_raw  # noqa: E402


def test_duplicate_listing_keys_do_not_require_timestamp_columns():
    rows = pd.DataFrame(
        [
            {"listing_key": "abc", "price": 20_000, "year": 2020,
             "make": "Honda", "model": "Civic", "mileage": 40_000},
            {"listing_key": "abc", "price": 20_500, "year": 2020,
             "make": "Honda", "model": "Civic", "mileage": 40_000},
        ]
    )

    cleaned, report = clean_raw(rows)

    assert len(cleaned) == 1
    assert report["dropped_duplicate_listing_key"] == 1


def test_missing_listing_keys_are_not_treated_as_duplicates():
    rows = pd.DataFrame(
        [
            {"listing_key": None, "price": 20_000, "year": 2020,
             "make": "Honda", "model": "Civic", "mileage": 40_000},
            {"listing_key": None, "price": 24_000, "year": 2021,
             "make": "Toyota", "model": "Camry", "mileage": 35_000},
        ]
    )

    cleaned, report = clean_raw(rows, reference_year=2026)

    assert len(cleaned) == 2
    assert report["dropped_duplicate_listing_key"] == 0
