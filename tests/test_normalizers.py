"""
Unit tests for the pure normalization / parsing helpers in preprocess_cars_data.py.

These functions turn messy, free-text scrape fields ("All-Wheel Drive", "20/28 mpg",
"3.5L V6") into the small, fixed vocabularies the model is trained on. They are pure
functions, so they're cheap to test and high-value as regression guards: a silent change
to one of them would quietly shift feature values for every future scrape.

Run:  python3 -m pytest tests/ -q
"""

import math
import os
import sys

import pandas as pd
import pytest

# Make the src/ package importable when pytest is run from anywhere.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from preprocess_cars_data import (  # noqa: E402
    normalize_accident,
    normalize_color,
    normalize_drivetrain,
    normalize_fuel,
    normalize_title_status,
    normalize_transmission,
    parse_engine_size_l,
    parse_mpg_from_text,
)


# --------------------------- color ---------------------------

@pytest.mark.parametrize("raw,expected", [
    ("Jet Black", "black"),
    ("Pearl White", "white"),
    ("Charcoal Gray", "gray"),
    ("Grey", "gray"),
    ("Brilliant Silver Metallic", "silver"),
    ("Navy Blue", "blue"),
    ("Crimson", "red"),
    ("Champagne", "brown"),
    ("Some Funky Marketing Name", "other"),
])
def test_normalize_color(raw, expected):
    assert normalize_color(raw) == expected


def test_normalize_color_missing():
    assert normalize_color(None) is None
    assert normalize_color(float("nan")) is None
    assert normalize_color("") is None


# --------------------------- drivetrain ---------------------------

@pytest.mark.parametrize("raw,expected", [
    ("All-Wheel Drive", "awd"),
    ("AWD", "awd"),
    ("Front Wheel Drive", "fwd"),
    ("FWD", "fwd"),
    ("Rear-Wheel Drive", "rwd"),
    ("4x4", "4wd"),
    ("Four Wheel Drive", "4wd"),
    ("nonsense", None),
])
def test_normalize_drivetrain(raw, expected):
    assert normalize_drivetrain(raw) == expected


# --------------------------- transmission ---------------------------

@pytest.mark.parametrize("raw,expected", [
    ("Continuously Variable Transmission (CVT)", "cvt"),
    ("6-Speed Manual", "manual"),
    ("8-Speed Automatic", "automatic"),
    ("Automatic", "automatic"),
    ("", None),
])
def test_normalize_transmission(raw, expected):
    assert normalize_transmission(raw) == expected


# --------------------------- fuel ---------------------------

@pytest.mark.parametrize("raw,expected", [
    ("Electric", "ev"),
    ("EV", "ev"),
    ("Gasoline Hybrid", "hybrid"),  # 'hybrid' is checked before 'gas'
    ("Diesel", "diesel"),
    ("Gasoline", "gas"),
    ("Regular Unleaded", "gas"),
    ("", None),
])
def test_normalize_fuel(raw, expected):
    assert normalize_fuel(raw) == expected


# --------------------------- title status ---------------------------

@pytest.mark.parametrize("raw,expected", [
    ("Clean", "clean"),
    ("Salvage Title", "salvage"),
    ("Rebuilt", "rebuilt"),
    (True, "clean"),
    (False, None),
    ("true", "clean"),
])
def test_normalize_title_status(raw, expected):
    assert normalize_title_status(raw) == expected


# --------------------------- accident history ---------------------------

@pytest.mark.parametrize("raw,expected", [
    ("No accidents reported", "no_accident"),
    ("At least 1 accident or damage reported", "had_accident"),
    (True, "no_accident"),
    (False, "had_accident"),
    ("true", "no_accident"),
    ("false", "had_accident"),
])
def test_normalize_accident(raw, expected):
    assert normalize_accident(raw) == expected


# --------------------------- engine size ---------------------------

@pytest.mark.parametrize("raw,expected", [
    ("3.5L V6", 3.5),
    ("2.0L Turbo I4", 2.0),
    ("5.0 L V8", 5.0),
    ("Electric Motor", None),
    (None, None),
])
def test_parse_engine_size_l(raw, expected):
    assert parse_engine_size_l(raw) == expected


# --------------------------- MPG ---------------------------

def test_parse_mpg_city_highway():
    out = parse_mpg_from_text("20 City / 28 Highway")
    assert out["mpg_city"] == 20
    assert out["mpg_highway"] == 28


def test_parse_mpg_single():
    out = parse_mpg_from_text("30 mpg combined")
    assert out["mpg_combined"] == 30


def test_parse_mpg_equivalent():
    out = parse_mpg_from_text("120 MPGe")
    assert out["mpg_equivalent"] == 120


def test_parse_mpg_missing():
    out = parse_mpg_from_text(None)
    assert all(v is None for v in out.values())
