from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .data import ENCODING_FEATURE


ENCODING_SMOOTHING = 10.0
MONOTONE_DIRECTIONS = {
    "year": 1,
    "mileage": -1,
    "owner_count": -1,
}

MODEL_PARAMETERS = {
    "n_estimators": 2_000,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
    "objective": "regression",
}


def compute_make_model_encoding(
    features: pd.DataFrame,
    log_target: pd.Series,
    smoothing: float = ENCODING_SMOOTHING,
) -> tuple[dict[str, float], float]:
    """Fit the smoothed make/model encoding on training rows only."""
    if "make_model" not in features.columns:
        raise ValueError("make_model is required for the price encoding")
    values = pd.Series(log_target).reset_index(drop=True).astype(float)
    groups = features["make_model"].reset_index(drop=True).astype(str)
    prior = float(values.mean())
    grouped = values.groupby(groups, observed=True)
    encoded = (grouped.sum() + prior * smoothing) / (grouped.count() + smoothing)
    return {str(key): float(value) for key, value in encoded.items()}, prior


def apply_make_model_encoding(
    features: pd.DataFrame,
    encoding: Mapping[str, float],
    prior: float,
) -> pd.DataFrame:
    result = features.copy()
    result[ENCODING_FEATURE] = (
        result["make_model"].astype(str).map(encoding).fillna(prior).astype(float)
    )
    return result


def monotone_constraints(columns: Sequence[str]) -> list[int]:
    return [MONOTONE_DIRECTIONS.get(column, 0) for column in columns]


def align_categories(
    train: pd.DataFrame,
    test: pd.DataFrame,
    categorical_features: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Use training-fold vocabularies and map unseen test values to missing."""
    train_aligned = train.copy()
    test_aligned = test.copy()
    active = [
        column for column in categorical_features if column in train_aligned.columns
    ]
    for column in active:
        train_values = train_aligned[column].astype("object")
        categories = pd.Index(train_values.dropna().unique())
        train_aligned[column] = pd.Categorical(train_values, categories=categories)
        test_values = test_aligned[column].astype("object")
        known = test_values.where(test_values.isin(categories))
        test_aligned[column] = pd.Categorical(known, categories=categories)
    return train_aligned, test_aligned, active


def fit_predict(
    train_features: pd.DataFrame,
    train_target: pd.Series,
    test_features: pd.DataFrame,
    categorical_features: Sequence[str],
    feature_order: Sequence[str],
    *,
    use_price_encoding: bool = True,
    smoke: bool = False,
) -> np.ndarray:
    """Fit the fixed log-price LightGBM model and return dollar predictions."""
    try:
        import lightgbm as lgb
    except ImportError as exc:  # pragma: no cover - exercised by installation checks
        raise RuntimeError("LightGBM is required to fit the experiment model") from exc

    x_train = train_features.reset_index(drop=True).copy()
    y_train = pd.Series(train_target).reset_index(drop=True).astype(float)
    x_test = test_features.reset_index(drop=True).copy()

    if use_price_encoding:
        encoding, prior = compute_make_model_encoding(x_train, np.log1p(y_train))
        x_train = apply_make_model_encoding(x_train, encoding, prior)
        x_test = apply_make_model_encoding(x_test, encoding, prior)
        selected = list(feature_order)
    else:
        selected = [column for column in feature_order if column != ENCODING_FEATURE]

    missing = sorted(set(selected).difference(x_train.columns))
    if missing:
        raise ValueError(
            f"model features missing from training frame: {', '.join(missing)}"
        )
    x_train = x_train[selected]
    x_test = x_test[selected]
    x_train, x_test, active_categories = align_categories(
        x_train, x_test, categorical_features
    )

    parameters = dict(MODEL_PARAMETERS)
    parameters["monotone_constraints"] = monotone_constraints(selected)
    if smoke:
        parameters.update({"n_estimators": 60, "n_jobs": 2})

    model = lgb.LGBMRegressor(**parameters)
    model.fit(
        x_train,
        np.log1p(y_train),
        categorical_feature=active_categories if active_categories else "auto",
    )
    return np.expm1(model.predict(x_test))
