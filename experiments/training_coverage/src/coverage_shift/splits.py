from __future__ import annotations

from collections import defaultdict, deque
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .data import BandSpec


def random_splits(n_rows: int, *, n_splits: int = 5, seed: int = 42):
    if n_rows < n_splits:
        raise ValueError("number of rows must be at least the number of folds")
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(np.arange(n_rows)))


def balanced_group_splits(
    groups: Sequence[object],
    *,
    n_splits: int = 5,
    seed: int = 42,
    trials: int = 500,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Make seeded, group-disjoint folds with approximately balanced row counts."""
    group_array = np.asarray(groups).astype(str)
    unique_groups, inverse, counts = np.unique(
        group_array, return_inverse=True, return_counts=True
    )
    if len(unique_groups) < n_splits:
        raise ValueError(
            f"grouped {n_splits}-fold validation needs at least {n_splits} distinct groups"
        )
    rng = np.random.default_rng(seed)
    best_score: tuple[int, float] | None = None
    best_assignment: np.ndarray | None = None
    for _ in range(trials):
        order = rng.permutation(len(unique_groups))
        loads = np.zeros(n_splits, dtype=int)
        assignment = np.full(len(unique_groups), -1, dtype=int)
        for group_position in order:
            lightest = np.flatnonzero(loads == loads.min())
            chosen_fold = int(rng.choice(lightest))
            assignment[group_position] = chosen_fold
            loads[chosen_fold] += int(counts[group_position])
        score = (int(loads.max() - loads.min()), float(loads.std()))
        if best_score is None or score < best_score:
            best_score = score
            best_assignment = assignment.copy()

    if best_assignment is None:
        raise AssertionError("group assignment failed")
    row_fold = best_assignment[inverse]
    all_rows = np.arange(len(group_array))
    return [
        (all_rows[row_fold != fold], all_rows[row_fold == fold])
        for fold in range(n_splits)
    ]


def standard_splits(
    kind: str,
    metadata: pd.DataFrame,
    *,
    n_splits: int = 5,
    seed: int = 42,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[str] | None]:
    if kind == "random":
        splits = random_splits(len(metadata), n_splits=n_splits, seed=seed)
        labels = None
    elif kind in {"make_model", "make"}:
        groups = metadata[kind].astype(str).to_numpy()
        splits = balanced_group_splits(groups, n_splits=n_splits, seed=seed)
        labels = [f"{kind}_groups"] * len(splits)
        for train_rows, test_rows in splits:
            if set(groups[train_rows]).intersection(groups[test_rows]):
                raise AssertionError(f"{kind} crosses a grouped fold")
    else:
        raise ValueError(f"unknown split kind: {kind}")

    assert_partition(splits, len(metadata))
    return splits, labels


def band_splits(
    metadata: pd.DataFrame,
    spec: BandSpec,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[str]]:
    group = metadata[spec.name].astype(str).to_numpy()
    all_rows = np.arange(len(metadata))
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    labels: list[str] = []
    for label in spec.labels:
        test_rows = all_rows[group == label]
        train_rows = all_rows[group != label]
        if len(test_rows) == 0:
            raise ValueError(f"range has no rows: {spec.name} / {label}")
        splits.append((train_rows, test_rows))
        labels.append(label)
    assert_partition(splits, len(metadata))
    return splits, labels


def assert_partition(
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    n_rows: int,
) -> None:
    counts = np.zeros(n_rows, dtype=int)
    for train_rows, test_rows in splits:
        if np.intersect1d(train_rows, test_rows).size:
            raise AssertionError("training and test indices overlap")
        counts[np.asarray(test_rows, dtype=int)] += 1
    if not np.all(counts == 1):
        raise AssertionError("test folds do not score every row exactly once")


def matched_outside_removals(
    represented_inside_rows: np.ndarray,
    outside_rows: np.ndarray,
    metadata: pd.DataFrame,
    *,
    seed: int,
) -> tuple[np.ndarray, dict[str, int | float | str]]:
    """Remove one outside row per represented inside row, matching composition."""
    represented_inside_rows = np.asarray(represented_inside_rows, dtype=int)
    outside_order = np.asarray(outside_rows, dtype=int).copy()
    if len(represented_inside_rows) > len(outside_order):
        raise ValueError("not enough outside rows for equal-size matching")

    rng = np.random.default_rng(seed)
    rng.shuffle(outside_order)
    available = set(int(row) for row in outside_order)
    by_model: dict[str, deque[int]] = defaultdict(deque)
    by_make: dict[str, deque[int]] = defaultdict(deque)
    all_rows: deque[int] = deque(int(row) for row in outside_order)
    for row in outside_order:
        by_model[str(metadata.iloc[row]["make_model"])].append(int(row))
        by_make[str(metadata.iloc[row]["make"])].append(int(row))

    def take(queue: deque[int]) -> int | None:
        while queue:
            candidate = queue.popleft()
            if candidate in available:
                available.remove(candidate)
                return candidate
        return None

    inside_order = represented_inside_rows.copy()
    rng.shuffle(inside_order)
    removed: list[int] = []
    exact_model = 0
    same_make = 0
    fallback = 0
    for row in inside_order:
        metadata_row = metadata.iloc[row]
        candidate = take(by_model[str(metadata_row["make_model"])])
        if candidate is not None:
            exact_model += 1
        else:
            candidate = take(by_make[str(metadata_row["make"])])
            if candidate is not None:
                same_make += 1
            else:
                candidate = take(all_rows)
                if candidate is None:
                    raise AssertionError("outside-row matching exhausted its pool")
                fallback += 1
        removed.append(candidate)

    import hashlib

    removed_array = np.array(sorted(removed), dtype=np.int64)
    digest = hashlib.sha256(removed_array.tobytes()).hexdigest()
    total = len(removed)
    return np.array(sorted(available), dtype=int), {
        "removed_rows": int(total),
        "same_make_model_matches": int(exact_model),
        "same_make_fallback_matches": int(same_make),
        "unrestricted_fallback_matches": int(fallback),
        "same_make_model_match_rate": float(exact_model / total if total else 0.0),
        "same_make_or_model_match_rate": float(
            (exact_model + same_make) / total if total else 0.0
        ),
        "removed_row_index_sha256": digest,
    }
