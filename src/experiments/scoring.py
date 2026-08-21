"""
Proper scoring rules for the probabilistic price predictions.

Coverage alone is a weak check: a band can hit 80% coverage by being absurdly
wide. Proper scoring rules reward calibration AND sharpness together, so they rank
the raw / marginal-CQR / Mondrian bands on the quality that actually matters.

All in dollar space (expm1 is monotone, so expm1(quantile_log) is a valid dollar
quantile), matching the coverage/width/MAE units used elsewhere. Lower is better.
"""

import numpy as np


def pinball_loss(y_true, q_pred, tau: float) -> float:
    """Mean pinball (quantile) loss for a predicted quantile at level tau.
    Minimized in expectation by the true tau-quantile - a proper scoring rule."""
    y_true = np.asarray(y_true, float)
    q_pred = np.asarray(q_pred, float)
    r = y_true - q_pred
    return float(np.mean(np.maximum(tau * r, (tau - 1) * r)))


def winkler_score(y_true, lo, hi, alpha: float = 0.20) -> float:
    """Mean Winkler interval score for a central (1-alpha) interval [lo, hi].

    Rewards narrow intervals but penalizes misses by 2/alpha times the shortfall -
    the proper scoring rule for interval forecasts.
    """
    y_true = np.asarray(y_true, float)
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    width = hi - lo
    below = lo - y_true
    above = y_true - hi
    penalty = np.where(y_true < lo, (2 / alpha) * below,
                       np.where(y_true > hi, (2 / alpha) * above, 0.0))
    return float(np.mean(width + penalty))


def score_band(y_true, lo, hi, q50=None, alpha: float = 0.20) -> dict:
    """Winkler score for the band, plus pinball at q10/q50/q90 when the edges/median
    are interpreted as those quantiles (alpha=0.2 -> q10/q90)."""
    tau_lo, tau_hi = alpha / 2, 1 - alpha / 2
    out = {
        "winkler": winkler_score(y_true, lo, hi, alpha),
        "pinball_q10": pinball_loss(y_true, lo, tau_lo),
        "pinball_q90": pinball_loss(y_true, hi, tau_hi),
    }
    if q50 is not None:
        out["pinball_q50"] = pinball_loss(y_true, q50, 0.50)
    return out
