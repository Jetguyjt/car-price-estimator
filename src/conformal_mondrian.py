"""
Mondrian (group-conditional) conformalized quantile regression.

Marginal CQR targets coverage for the q10-q90 band on average. That average can
hide differences across price ranges when prediction errors change with vehicle
price. This module partitions the calibration set by predicted price and computes
a separate adjustment per bin.

The goal is better empirical balance across price segments. It is not a guarantee
of coverage for every segment or future dataset. Sparse groups borrow the global
adjustment instead of estimating an unstable group-specific value.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

import conformal

ARTIFACT_NAME = "conformal_mondrian.json"
MIN_GROUP_N = 40  # below this, a bin borrows the global delta rather than self-calibrate


def _bin_edges_from_predictions(pred_log: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile bin edges on predicted log-price. Interior edges only; the outer
    edges are -inf/+inf so any future prediction lands in a bin."""
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    interior = np.unique(np.quantile(pred_log, qs))
    return np.concatenate([[-np.inf], interior, [np.inf]])


def assign_bins(pred_log: Sequence[float], edges: Sequence[float]) -> np.ndarray:
    """Map predictions to bin indices [0, n_bins-1] using np.digitize."""
    edges = np.asarray(edges, dtype=float)
    return np.clip(np.digitize(np.asarray(pred_log, dtype=float), edges[1:-1]),
                   0, len(edges) - 2).astype(int)


def fit(lo_log, hi_log, mid_log, y_log, *, target_coverage: float = 0.80,
        n_bins: int = 5) -> Dict:
    """Compute per-bin CQR deltas keyed on predicted (mid/q50) log-price.

    lo_log/hi_log: calibration q10/q90 predictions (log space).
    mid_log: calibration q50 (or point) predictions, used only to assign bins.
    y_log: calibration actuals (log space).
    """
    lo_log = np.asarray(lo_log, float)
    hi_log = np.asarray(hi_log, float)
    mid_log = np.asarray(mid_log, float)
    y_log = np.asarray(y_log, float)

    global_delta = conformal.compute_log_delta(lo_log, hi_log, y_log, target_coverage)
    edges = _bin_edges_from_predictions(mid_log, n_bins)
    bins = assign_bins(mid_log, edges)

    deltas: List[float] = []
    counts: List[int] = []
    for b in range(len(edges) - 1):
        mask = bins == b
        counts.append(int(mask.sum()))
        if mask.sum() < MIN_GROUP_N:
            deltas.append(global_delta)
        else:
            deltas.append(conformal.compute_log_delta(
                lo_log[mask], hi_log[mask], y_log[mask], target_coverage))

    return {
        "method": "Mondrian conformalized quantile regression (bin key: predicted q50)",
        "target_coverage": target_coverage,
        "n_bins": n_bins,
        "bin_edges_log": [float(e) for e in edges],
        "bin_deltas_log": [float(d) for d in deltas],
        "bin_counts": counts,
        "global_delta_log": float(global_delta),
        "calibration_n": int(len(y_log)),
        "min_group_n": MIN_GROUP_N,
    }


def deltas_for(model: Dict, mid_log: Sequence[float]):
    """Per-point (lo_delta, hi_delta) arrays for test predictions.

    Both band edges share a bin's delta; returned separately so callers can apply
    q10_log - lo_delta and q90_log + hi_delta symmetrically.
    """
    edges = model["bin_edges_log"]
    bin_deltas = np.asarray(model["bin_deltas_log"], float)
    bins = assign_bins(mid_log, edges)
    d = bin_deltas[bins]
    return d, d


def save(artifact_dir, model: Dict) -> None:
    (Path(artifact_dir) / ARTIFACT_NAME).write_text(json.dumps(model, indent=2))


def load(artifact_dir) -> Optional[Dict]:
    path = Path(artifact_dir) / ARTIFACT_NAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError):
        return None
