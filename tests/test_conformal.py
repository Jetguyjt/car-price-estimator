"""
Tests for the conformal (CQR) band calibration in src/conformal.py.

These guard the coverage guarantee: after calibration, a held-out set should be
covered at ~the target rate whether the raw band starts too narrow or too wide.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import conformal  # noqa: E402


def _band(rng, n, half_width, err_std=1.0):
    """Realistic band: an interval of +/- half_width around a *prediction* that
    differs from the truth y by a random error. Coverage depends on how the band
    half-width compares to the prediction error spread."""
    y = rng.normal(10.0, 1.0, n)
    center = y + rng.normal(0.0, err_std, n)  # model prediction, off by some error
    return y, center - half_width, center + half_width


def _coverage(lo, hi, y, delta):
    return float(np.mean((y >= lo - delta) & (y <= hi + delta)))


def test_widens_a_too_narrow_band_to_target():
    rng = np.random.default_rng(0)
    y, lo, hi = _band(rng, 4000, half_width=0.5)   # narrow vs err_std=1.0 -> under-covers
    assert _coverage(lo, hi, y, 0.0) < 0.60        # sanity: raw band really is too narrow
    delta = conformal.compute_log_delta(lo, hi, y, target_coverage=0.80)
    assert delta > 0                                # must widen
    assert 0.77 <= _coverage(lo, hi, y, delta) <= 0.83


def test_tightens_a_too_wide_band_to_target():
    rng = np.random.default_rng(1)
    y, lo, hi = _band(rng, 4000, half_width=3.0)   # wide vs err_std=1.0 -> over-covers
    assert _coverage(lo, hi, y, 0.0) > 0.95        # sanity: raw band really is too wide
    delta = conformal.compute_log_delta(lo, hi, y, target_coverage=0.80)
    assert delta < 0                                # must tighten
    assert 0.77 <= _coverage(lo, hi, y, delta) <= 0.83


def test_higher_target_never_gives_smaller_delta():
    rng = np.random.default_rng(2)
    y, lo, hi = _band(rng, 2000, half_width=0.5)
    d80 = conformal.compute_log_delta(lo, hi, y, 0.80)
    d90 = conformal.compute_log_delta(lo, hi, y, 0.90)
    assert d90 >= d80  # a stricter coverage target needs an at-least-as-wide band


def test_empty_calibration_set_is_safe():
    assert conformal.compute_log_delta([], [], [], 0.80) == 0.0


def test_load_delta_missing_artifact_returns_zero(tmp_path):
    assert conformal.load_delta(tmp_path) == 0.0


def test_save_then_load_roundtrip(tmp_path):
    conformal.save(tmp_path, 0.0451, 0.80, 2019)
    assert abs(conformal.load_delta(tmp_path) - 0.0451) < 1e-9
