"""
Tests for the conformal (CQR) band calibration in src/conformal.py.

These checks verify how the calibration adjustment behaves on synthetic calibration
samples. They do not establish future coverage; that must be measured on separate
evaluation data.
"""

import os
import sys
import json

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


def test_crossed_quantiles_are_ordered_before_calibration():
    y = np.array([10.0, 10.5, 11.0])
    crossed_lo = np.array([11.0, 11.0, 11.5])
    crossed_hi = np.array([9.0, 10.0, 10.5])

    crossed = conformal.compute_log_delta(crossed_lo, crossed_hi, y, 0.80)
    ordered = conformal.compute_log_delta(crossed_hi, crossed_lo, y, 0.80)

    assert crossed == ordered


def test_display_median_never_expands_calibrated_endpoints():
    lo, mid, hi = conformal.calibrated_band_logs(
        q10_log=np.array([9.0, 11.0]),
        q50_log=np.array([12.0, 8.0]),
        q90_log=np.array([11.0, 9.0]),
        lo_delta=np.array([0.2, 0.2]),
        hi_delta=np.array([0.3, 0.3]),
    )

    np.testing.assert_allclose(lo, [8.8, 8.8])
    np.testing.assert_allclose(hi, [11.3, 11.3])
    np.testing.assert_allclose(mid, [11.3, 8.8])


# --------------------- Mondrian group-conditional CQR ---------------------

import numpy as np  # noqa: E402
import conformal_mondrian as cm  # noqa: E402


def _synth(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    mid = rng.uniform(8, 12, n)
    spread = 0.05 + 0.15 * (mid - 8) / 4          # heteroscedastic: wider at the top
    y = mid + rng.normal(0, spread)
    lo = mid - 0.10                                # fixed-width raw band
    hi = mid + 0.10
    return lo, hi, mid, y


def test_mondrian_fixes_conditional_coverage():
    lo, hi, mid, y = _synth()
    model = cm.fit(lo, hi, mid, y, target_coverage=0.80, n_bins=5)
    lo_d, hi_d = cm.deltas_for(model, mid)
    overall = np.mean((y >= lo - lo_d) & (y <= hi + hi_d))
    assert abs(overall - 0.80) < 0.03
    # top segment: marginal under-covers, Mondrian repairs it
    import conformal
    g = conformal.compute_log_delta(lo, hi, y, 0.80)
    top = mid > np.quantile(mid, 0.8)
    marg_top = np.mean((y[top] >= lo[top] - g) & (y[top] <= hi[top] + g))
    mond_top = np.mean((y[top] >= lo[top] - lo_d[top]) & (y[top] <= hi[top] + hi_d[top]))
    assert mond_top > marg_top + 0.10


def test_mondrian_deltas_increase_with_predicted_price():
    lo, hi, mid, y = _synth()
    model = cm.fit(lo, hi, mid, y, target_coverage=0.80, n_bins=5)
    deltas = model["bin_deltas_log"]
    assert deltas[-1] > deltas[0]                  # wider band needed for pricier cars


def test_mondrian_small_group_borrows_global():
    lo, hi, mid, y = _synth(n=60)                  # < MIN_GROUP_N per bin
    model = cm.fit(lo, hi, mid, y, target_coverage=0.80, n_bins=10)
    g = model["global_delta_log"]
    assert all(abs(d - g) < 1e-9 for d in model["bin_deltas_log"])


def test_mondrian_assign_bins_covers_extremes():
    edges = [-np.inf, 9.0, 10.0, 11.0, np.inf]
    bins = cm.assign_bins([-100.0, 9.5, 10.5, 500.0], edges)
    assert list(bins) == [0, 1, 2, 3]


def test_mondrian_save_load_roundtrip(tmp_path):
    lo, hi, mid, y = _synth()
    model = cm.fit(lo, hi, mid, y, target_coverage=0.80, n_bins=5)
    cm.save(tmp_path, model)
    artifact_text = (tmp_path / cm.ARTIFACT_NAME).read_text()
    json.loads(
        artifact_text,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-standard JSON constant: {value}")
        ),
    )
    assert "Infinity" not in artifact_text
    loaded = cm.load(tmp_path)
    assert loaded["bin_deltas_log"] == model["bin_deltas_log"]
    assert cm.load(tmp_path / "nonexistent") is None
