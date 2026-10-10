import numpy as np
import pandas as pd

from coverage_shift.splits import (
    assert_partition,
    balanced_group_splits,
    matched_outside_removals,
)


def test_balanced_group_splits_are_reproducible_and_group_disjoint():
    groups = np.repeat(
        [f"group-{index}" for index in range(12)], [3, 7, 5, 8, 2, 6, 4, 9, 3, 5, 7, 4]
    )
    first = balanced_group_splits(groups, seed=42)
    second = balanced_group_splits(groups, seed=42)

    assert_partition(first, len(groups))
    assert [test.tolist() for _, test in first] == [test.tolist() for _, test in second]
    for train, test in first:
        assert set(groups[train]).isdisjoint(groups[test])
    fold_sizes = [len(test) for _, test in first]
    assert max(fold_sizes) - min(fold_sizes) <= 4


def test_matched_removal_keeps_equal_training_size_and_tracks_fallbacks():
    metadata = pd.DataFrame(
        {
            "make": ["a", "a", "b", "b", "a", "a", "b", "c", "d", "e"],
            "make_model": ["a1", "a2", "b1", "b2", "a1", "a3", "b1", "c1", "d1", "e1"],
        }
    )
    represented_inside = np.array([0, 1, 2])
    outside = np.arange(3, 10)

    remaining, diagnostics = matched_outside_removals(
        represented_inside, outside, metadata, seed=123
    )

    assert len(remaining) == len(outside) - len(represented_inside)
    assert diagnostics["removed_rows"] == len(represented_inside)
    assert diagnostics["same_make_model_matches"] + diagnostics[
        "same_make_fallback_matches"
    ] + diagnostics["unrestricted_fallback_matches"] == len(represented_inside)
    assert set(remaining).issubset(set(outside))
