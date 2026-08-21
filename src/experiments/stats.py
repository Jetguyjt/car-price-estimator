"""
Statistical-rigor utilities: bootstrap confidence intervals for error metrics
and a paired bootstrap test for comparing two models on the same test set.

Point estimates alone ("MAE $2,947") do not show how much a reported metric
varies under resampling. These helpers add intervals to selected error metrics and a
paired bootstrap comparison for models scored on the same rows. Other experiments
report their own descriptive checks.
"""

from typing import Callable, Dict, Tuple

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

DEFAULT_B = 2000
RNG_SEED = 42


def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), 1e-9)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


METRICS: Dict[str, Callable] = {
    "MAE": lambda yt, yp: float(mean_absolute_error(yt, yp)),
    "RMSE": lambda yt, yp: float(np.sqrt(np.mean((np.asarray(yt) - np.asarray(yp)) ** 2))),
    "R2": lambda yt, yp: float(r2_score(yt, yp)),
    "MAPE": mape,
}


def bootstrap_ci(y_true, y_pred, metric: str = "MAE", B: int = DEFAULT_B,
                 seed: int = RNG_SEED, alpha: float = 0.05) -> Tuple[float, float, float]:
    """Return (point, lo, hi) for a metric via case-resampling bootstrap."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    fn = METRICS[metric]
    point = fn(y_true, y_pred)
    n = len(y_true)
    rng = np.random.default_rng(seed)
    stats = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        stats[b] = fn(y_true[idx], y_pred[idx])
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def paired_bootstrap_test(y_true, pred_a, pred_b, metric: str = "MAE",
                          B: int = DEFAULT_B, seed: int = RNG_SEED) -> Dict:
    """Paired bootstrap on the metric difference (A - B) over shared test rows.

    Returns the observed difference, its 95% CI, and a two-sided p-value for
    H0: no difference. Lower-is-better metrics (MAE/RMSE/MAPE) => negative diff
    means A beats B.
    """
    y_true = np.asarray(y_true, dtype=float)
    pred_a = np.asarray(pred_a, dtype=float)
    pred_b = np.asarray(pred_b, dtype=float)
    fn = METRICS[metric]
    observed = fn(y_true, pred_a) - fn(y_true, pred_b)
    n = len(y_true)
    rng = np.random.default_rng(seed)
    diffs = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        diffs[b] = fn(y_true[idx], pred_a[idx]) - fn(y_true[idx], pred_b[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # Two-sided p: fraction of bootstrap diffs on the opposite side of 0 from observed.
    if observed <= 0:
        p = 2 * np.mean(diffs >= 0)
    else:
        p = 2 * np.mean(diffs <= 0)
    return {"metric": metric, "diff": float(observed), "ci_lo": float(lo),
            "ci_hi": float(hi), "p_value": float(min(1.0, p))}


def fmt_ci(point, lo, hi, dollar: bool = False, pct: bool = False) -> str:
    if dollar:
        return f"${point:,.0f} [{lo:,.0f}, {hi:,.0f}]"
    if pct:
        return f"{point:.1f}% [{lo:.1f}, {hi:.1f}]"
    return f"{point:.3f} [{lo:.3f}, {hi:.3f}]"
