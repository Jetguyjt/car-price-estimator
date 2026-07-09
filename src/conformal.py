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

By the CQR theorem this covers the target fraction (e.g. 80%) of future points, given
exchangeability. `delta` may be negative if the raw band over-covers (it would tighten).

Everything is stored in `model_artifacts/conformal.json` and applied identically at
train-time evaluation, in `evaluate_model.py`, and in `predict_listing.py`, so reported
coverage and served bands agree. If the artifact is absent, `load_delta` returns 0.0 and
the pipeline behaves exactly as before (backward compatible with pre-v4 models).
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

    scores = np.maximum(lo_log - y_log, y_log - hi_log)
    n = scores.size
    if n == 0:
        return 0.0

    # Finite-sample rank: the ceil((n+1)*coverage)-th smallest score. If that rank
    # exceeds n (target so high the sample can't certify it), fall back to the max
    # score, which is the most conservative (widest) choice available.
    rank = math.ceil((n + 1) * target_coverage)
    rank = min(rank, n)
    return float(np.sort(scores)[rank - 1])


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
