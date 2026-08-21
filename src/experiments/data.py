"""
Single source of truth for the experiment data split.

Reproduces train_price_model's exact preprocessing and 3-way split
(proper-train / calibration / test) with the same seed, so every study in this
package trains and scores on identical rows - the precondition for a paper's
ablation and model-comparison tables to be comparable.

The make_model target encoding is fit on the proper-train slice only (no leakage
into calibration or test), matching the production pipeline. Test-row metadata
(actual price, make, luxury flag, price bracket) is carried alongside for the
conditional-coverage analysis.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import paths
import train_price_model as tpm

RANDOM_SEED = tpm.RANDOM_SEED
TEST_SIZE = tpm.TEST_SIZE
CALIBRATION_FRACTION = tpm.CALIBRATION_FRACTION

# Fixed price brackets used everywhere conditional coverage / segment error is
# reported, so every figure and table bins the same way.
PRICE_BRACKETS = [0, 15_000, 30_000, 50_000, 80_000, np.inf]
PRICE_BRACKET_LABELS = ["<15k", "15-30k", "30-50k", "50-80k", "80k+"]


def price_bracket(prices) -> pd.Categorical:
    return pd.cut(prices, bins=PRICE_BRACKETS, labels=PRICE_BRACKET_LABELS)


@dataclass
class ExperimentData:
    # feature frames (with make_model_price_enc appended, matching production)
    x_ptrain: pd.DataFrame
    x_cal: pd.DataFrame
    x_train: pd.DataFrame          # ptrain + cal, for models that don't need a cal split
    x_test: pd.DataFrame
    # raw-dollar targets
    y_ptrain: pd.Series
    y_cal: pd.Series
    y_train: pd.Series
    y_test: pd.Series
    cat_cols: List[str]
    feature_names: List[str]
    # aligned test metadata for conditional analysis
    test_meta: pd.DataFrame        # columns: make, actual, price_bracket, is_luxury

    # log-space helpers
    @property
    def y_ptrain_log(self):
        return np.log1p(self.y_ptrain)

    @property
    def y_cal_log(self):
        return np.log1p(self.y_cal)

    @property
    def y_train_log(self):
        return np.log1p(self.y_train)

    @property
    def y_test_log(self):
        return np.log1p(self.y_test)


def load(with_encoding: bool = True) -> ExperimentData:
    """Build the canonical experiment split from the processed dataset."""
    df = tpm.load_dataset(tpm.DATA_PATH)
    x, y, feature_names, cat_cols = tpm.prepare_features(df)

    # Reconstruct per-row make/luxury metadata aligned to x (prepare_features drops
    # the same rows, in order, that we re-derive here).
    cleaned = df.copy().dropna(subset=[tpm.TARGET_COLUMN]).reset_index(drop=True)
    yy = pd.to_numeric(cleaned[tpm.TARGET_COLUMN], errors="coerce")
    valid = yy.notna() & (yy > 0)
    meta_all = cleaned.loc[valid, ["make"]].reset_index(drop=True)
    if "is_luxury_make" in cleaned.columns:
        meta_all["is_luxury"] = cleaned.loc[valid, "is_luxury_make"].reset_index(drop=True)
    else:
        meta_all["is_luxury"] = 0
    assert len(meta_all) == len(x)

    x_train, x_test, y_train, y_test, m_train, m_test = train_test_split(
        x, y, meta_all, test_size=TEST_SIZE, random_state=RANDOM_SEED,
    )
    x_ptrain, x_cal, y_ptrain, y_cal = train_test_split(
        x_train, y_train, test_size=CALIBRATION_FRACTION, random_state=RANDOM_SEED,
    )

    if with_encoding:
        enc, prior = tpm.compute_make_model_encoding(x_ptrain, np.log1p(y_ptrain))
        x_ptrain = tpm.apply_make_model_encoding(x_ptrain, enc, prior)
        x_cal = tpm.apply_make_model_encoding(x_cal, enc, prior)
        x_train = tpm.apply_make_model_encoding(x_train, enc, prior)
        x_test = tpm.apply_make_model_encoding(x_test, enc, prior)
        feature_names = feature_names + [tpm.ENCODING_FEATURE]

    test_meta = m_test.copy()
    test_meta["actual"] = y_test.to_numpy()
    test_meta["price_bracket"] = price_bracket(y_test.to_numpy())
    test_meta = test_meta.reset_index(drop=True)

    return ExperimentData(
        x_ptrain=x_ptrain, x_cal=x_cal, x_train=x_train, x_test=x_test,
        y_ptrain=y_ptrain, y_cal=y_cal, y_train=y_train, y_test=y_test,
        cat_cols=cat_cols, feature_names=feature_names, test_meta=test_meta,
    )
