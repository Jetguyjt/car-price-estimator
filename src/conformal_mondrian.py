"""
Mondrian (group-conditional) conformalized quantile regression.

Marginal CQR (see conformal.py) targets coverage for the q10-q90 band *on average*.
That average can hide substantial differences across price ranges when prediction
errors change with vehicle price. A single global adjustment cannot directly account
for those differences.

This implementation partitions the calibration set by predicted price and computes
a separate CQR adjustment per bin. A test point receives its bin's adjustment. The
goal is better empirical balance across price segments; it is not a guarantee of
coverage for every segment or future dataset.

The group must be knowable at prediction time, so we key it off the model's own
*predicted* price (q50), not the actual price - binning by prediction is legitimate
(no target leakage) and puts each car in the segment the model thinks it belongs to.
An optional categorical key (e.g. is_luxury_make) is also supported.

Groups with too few calibration points borrow the global (marginal) delta instead
of estimating an unstable group-specific adjustment.
Artifact format is a superset of conformal.json; load_delta-style helpers keep the
serving path simple.
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


def _json_edges(edges: Sequence[float]) -> List[float | str]:
    """Use JSON strings for infinite outer bounds.

    Python's JSON encoder otherwise writes bare Infinity values, which are accepted
    by Python but are not part of the JSON standard. NumPy accepts the strings
    ``-inf`` and ``inf`` when the artifact is converted back to floats.
    """
    output: List[float | str] = []
    for edge in edges:
        value = float(edge)
        if np.isneginf(value):
            output.append("-inf")
        elif np.isposinf(value):
            output.append("inf")
        else:
            output.append(value)
    return output


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
        "bin_edges_log": _json_edges(edges),
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
    (Path(artifact_dir) / ARTIFACT_NAME).write_text(
        json.dumps(model, indent=2, allow_nan=False) + "\n"
    )


def load(artifact_dir) -> Optional[Dict]:
    path = Path(artifact_dir) / ARTIFACT_NAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError):
        return None
