import numpy as np
import pytest

from coverage_shift.metrics import metric_record, paired_cluster_bootstrap


def test_metrics_use_dollar_and_percentage_scales():
    result = metric_record([10_000, 20_000], [9_000, 24_000])

    assert result["mae"] == 2_500
    assert result["mape_percent"] == pytest.approx(15)
    assert result["mdape_percent"] == pytest.approx(15)
    assert result["wape_percent"] == pytest.approx(100 * 5_000 / 30_000)


def test_paired_cluster_bootstrap_is_deterministic():
    reference = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    shifted = reference + np.array([1, 1, 1, 2, 2, 2], dtype=float)
    groups = ["a", "a", "b", "b", "c", "c"]

    first = paired_cluster_bootstrap(reference, shifted, groups, seed=42, reps=200)
    second = paired_cluster_bootstrap(reference, shifted, groups, seed=42, reps=200)

    assert first == second
    assert first["delta_ci_low"] > 0
    assert first["ratio_ci_low"] > 1
    assert first["n_clusters"] == 3
