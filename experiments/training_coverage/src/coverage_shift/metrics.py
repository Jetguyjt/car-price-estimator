from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


def metric_record(
    actual: Sequence[float], predicted: Sequence[float]
) -> dict[str, float | int]:
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    if actual_array.shape != predicted_array.shape:
        raise ValueError("actual and predicted values must have the same shape")
    if actual_array.size == 0:
        raise ValueError("metrics require at least one row")
    if not np.all(np.isfinite(actual_array)) or not np.all(actual_array > 0):
        raise ValueError("actual prices must be positive and finite")
    if not np.all(np.isfinite(predicted_array)):
        raise ValueError("predictions must be finite")

    error = predicted_array - actual_array
    absolute_error = np.abs(error)
    percentage_error = absolute_error / actual_array
    return {
        "n": int(actual_array.size),
        "mae": float(absolute_error.mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "r2": float(r2_score(actual_array, predicted_array)),
        "mape_percent": float(100 * percentage_error.mean()),
        "mdape_percent": float(100 * np.median(percentage_error)),
        "wape_percent": float(100 * absolute_error.sum() / actual_array.sum()),
    }


def paired_cluster_bootstrap(
    reference_error: Sequence[float],
    shifted_error: Sequence[float],
    clusters: Sequence[object],
    *,
    seed: int,
    reps: int = 2_000,
) -> dict[str, float | int]:
    """Bootstrap paired losses by resampling whole groups with replacement."""
    reference = np.asarray(reference_error, dtype=float)
    shifted = np.asarray(shifted_error, dtype=float)
    cluster_values = np.asarray(clusters).astype(str)
    if not (len(reference) == len(shifted) == len(cluster_values)):
        raise ValueError("paired losses and cluster labels must have equal length")
    if len(reference) == 0:
        raise ValueError("bootstrap requires at least one paired loss")
    if reps < 1:
        raise ValueError("bootstrap reps must be positive")

    frame = pd.DataFrame(
        {"reference": reference, "shifted": shifted, "cluster": cluster_values}
    )
    grouped = [
        group[["reference", "shifted"]].to_numpy(float)
        for _, group in frame.groupby("cluster", sort=False)
    ]
    rng = np.random.default_rng(seed)
    deltas = np.empty(reps, dtype=float)
    ratios = np.empty(reps, dtype=float)
    for index in range(reps):
        selected = rng.integers(0, len(grouped), size=len(grouped))
        sample = np.concatenate([grouped[position] for position in selected], axis=0)
        reference_mean = sample[:, 0].mean()
        shifted_mean = sample[:, 1].mean()
        deltas[index] = shifted_mean - reference_mean
        ratios[index] = shifted_mean / reference_mean

    if not np.all(np.isfinite(ratios)):
        raise ValueError("bootstrap ratio is undefined because reference error is zero")
    return {
        "delta_ci_low": float(np.quantile(deltas, 0.025)),
        "delta_ci_high": float(np.quantile(deltas, 0.975)),
        "ratio_ci_low": float(np.quantile(ratios, 0.025)),
        "ratio_ci_high": float(np.quantile(ratios, 0.975)),
        "bootstrap_reps": int(reps),
        "n_clusters": int(len(grouped)),
    }
