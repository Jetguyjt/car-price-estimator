"""
Conformalized Quantile Regression (CQR) adjustment for the price band.

The raw LightGBM quantile models (q10/q90) are trained independently and are not
guaranteed to be calibrated: at v3 the nominal-80% band only covered ~63% of held-out
prices. CQR fixes this with a distribution-free, finite-sample coverage guarantee.

Given a calibration set the models never trained on, we measure the conformity score

    E_i = max(q10_log(x_i) - y_log_i,  y_log_i - q90_log(x_i))

i.e. how far outside the predicted band each point falls (in log1p-price space; negative
when the point is comfortably inside). The additive adjustment `delta` is the
finite-sample-corrected empirical quantile of those scores, and the calibrated band is

    [ expm1(q10_log - delta),  expm1(q90_log + delta) ]

Under the usual exchangeability assumptions, this targets the requested future
coverage. `delta` may be negative if the raw band over-covers (it would tighten).

The global adjustment is stored in `model_artifacts/conformal.json`. Newer models
prefer price-bin adjustments from `conformal_mondrian.json`; the global value remains
the fallback for older artifacts. Training, evaluation, and serving use the same
selected adjustment. If the global artifact is absent, `load_delta` returns 0.0.
"""

import json
import math
from pathlib import Path

import numpy as np

ARTIFACT_NAME = "conformal.json"


def compute_log_delta(lo_log, hi_log, y_log, target_coverage: float = 0.80) -> float:
    """CQR additive adjustment (in log space) for a target interval coverage.

    lo_log / hi_log: model predictions for the lower (q10) and upper (q90) quantiles
    on the calibration set, in log1p(price) space.
    y_log: true log1p(price) on the calibration set.
    """
    lo_log = np.asarray(lo_log, dtype=float)
    hi_log = np.asarray(hi_log, dtype=float)
    y_log = np.asarray(y_log, dtype=float)

    # Independently trained quantile models can cross. Their two predictions still
    # define endpoints, so order them before computing conformity scores.
    ordered_lo = np.minimum(lo_log, hi_log)
    ordered_hi = np.maximum(lo_log, hi_log)
    scores = np.maximum(ordered_lo - y_log, y_log - ordered_hi)
    n = scores.size
    if n == 0:
        return 0.0

    # Finite-sample rank: the ceil((n+1)*coverage)-th smallest score. If that rank
    # exceeds n (target so high the sample can't certify it), fall back to the max
    # score, which is the most conservative (widest) choice available.
    rank = math.ceil((n + 1) * target_coverage)
    rank = min(rank, n)
    return float(np.sort(scores)[rank - 1])


def calibrated_band_logs(q10_log, q50_log, q90_log, lo_delta, hi_delta=None):
    """Return ordered calibrated endpoints and a median clamped inside them.

    q50 is displayed inside the interval but never allowed to expand it. This keeps
    the served endpoints identical to the endpoints that conformal calibration
    actually adjusted.
    """
    q10_log = np.asarray(q10_log, dtype=float)
    q50_log = np.asarray(q50_log, dtype=float)
    q90_log = np.asarray(q90_log, dtype=float)
    lo_delta = np.asarray(lo_delta, dtype=float)
    hi_delta = lo_delta if hi_delta is None else np.asarray(hi_delta, dtype=float)

    raw_lo = np.minimum(q10_log, q90_log)
    raw_hi = np.maximum(q10_log, q90_log)
    adjusted_a = raw_lo - lo_delta
    adjusted_b = raw_hi + hi_delta
    band_lo = np.minimum(adjusted_a, adjusted_b)
    band_hi = np.maximum(adjusted_a, adjusted_b)
    median = np.clip(q50_log, band_lo, band_hi)
    return band_lo, median, band_hi


def load_delta(artifact_dir) -> float:
    """Return the saved log-space adjustment, or 0.0 if none exists."""
    path = Path(artifact_dir) / ARTIFACT_NAME
    if not path.exists():
        return 0.0
    try:
        return float(json.loads(path.read_text()).get("log_delta", 0.0))
    except (ValueError, OSError):
        return 0.0


def save(artifact_dir, delta: float, target_coverage: float, n: int) -> None:
    path = Path(artifact_dir) / ARTIFACT_NAME
    path.write_text(json.dumps({
        "log_delta": delta,
        "target_coverage": target_coverage,
        "calibration_n": int(n),
        "method": "conformalized quantile regression (CQR)",
        "note": ("Additive adjustment in log1p(price) space: "
                 "band = [expm1(q10_log - log_delta), expm1(q90_log + log_delta)]."),
    }, indent=2))
