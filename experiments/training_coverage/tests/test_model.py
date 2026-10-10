import numpy as np
import pandas as pd

from coverage_shift.model import (
    MODEL_PARAMETERS,
    align_categories,
    apply_make_model_encoding,
    compute_make_model_encoding,
    monotone_constraints,
)


def test_make_model_encoding_is_smoothed_and_unseen_values_use_prior():
    train = pd.DataFrame({"make_model": ["a__one", "a__one", "b__one"]})
    log_price = pd.Series(np.log1p([10_000, 12_000, 30_000]))
    encoding, prior = compute_make_model_encoding(train, log_price)
    test = pd.DataFrame({"make_model": ["a__one", "unseen__line"]})

    transformed = apply_make_model_encoding(test, encoding, prior)

    assert transformed.loc[0, "make_model_price_enc"] == encoding["a__one"]
    assert transformed.loc[1, "make_model_price_enc"] == prior
    assert min(log_price) < encoding["a__one"] < max(log_price)


def test_category_vocabulary_comes_only_from_training_rows():
    train = pd.DataFrame({"trim": ["base", "sport"], "mileage": [10, 20]})
    test = pd.DataFrame({"trim": ["base", "new-trim"], "mileage": [30, 40]})

    train_aligned, test_aligned, active = align_categories(train, test, ["trim"])

    assert active == ["trim"]
    assert list(train_aligned["trim"].cat.categories) == ["base", "sport"]
    assert test_aligned.loc[0, "trim"] == "base"
    assert pd.isna(test_aligned.loc[1, "trim"])


def test_fixed_model_settings_and_monotonicity():
    assert MODEL_PARAMETERS["n_estimators"] == 2_000
    assert MODEL_PARAMETERS["learning_rate"] == 0.05
    assert MODEL_PARAMETERS["num_leaves"] == 63
    assert MODEL_PARAMETERS["min_child_samples"] == 20
    assert monotone_constraints(["year", "mileage", "owner_count", "trim_tier"]) == [
        1,
        -1,
        -1,
        0,
    ]
