"""
Model-fitting helpers shared by the experiment runners.

`fit_point` trains a single point model (the LightGBM mean model by default, or a
named sklearn comparator) on a chosen feature subset and returns test-set dollar
predictions - the unit the ablation and model-comparison studies consume.

`fit_bundle` trains the full quantile trio (q10/q50/q90) plus the mean model the
way production does, and returns both test and calibration predictions in log
space, which the conformal study needs to fit marginal and Mondrian adjustments.
"""

from typing import Dict, List, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

import train_price_model as tpm
from experiments.data import ExperimentData

# One fixed hyperparameter set for all LightGBM fits in the paper experiments, so
# ablation differences reflect features/method, not incidental HP-search noise.
LGBM_PARAMS = {"num_leaves": 63, "learning_rate": 0.05, "min_child_samples": 20}


def _prep(x: pd.DataFrame, features: List[str], cat_cols: List[str]):
    cols = [c for c in features if c in x.columns]
    sub = x[cols].copy()
    cats = [c for c in cat_cols if c in cols]
    return sub, cats


def fit_point(data: ExperimentData, features: List[str], *, log_target: bool = True,
              model: str = "lgbm", monotone: bool = False) -> np.ndarray:
    """Return dollar-space test predictions for a point model on `features`.

    monotone=True applies the production monotone constraints (LightGBM only).
    """
    x_tr, cats = _prep(data.x_train, features, data.cat_cols)
    x_te, _ = _prep(data.x_test, features, data.cat_cols)
    y_tr = data.y_train_log if log_target else data.y_train

    if model == "lgbm":
        params = dict(LGBM_PARAMS)
        if monotone:
            params["monotone_constraints"] = tpm.monotone_constraints_for(x_tr.columns.tolist())
        m = tpm.make_lgbm("regression", params=params)
        m.fit(x_tr, y_tr, categorical_feature=cats if cats else "auto")
        pred = m.predict(x_te)
    else:
        pred = _fit_sklearn(model, x_tr, y_tr, x_te, cats)

    return np.expm1(pred) if log_target else np.asarray(pred)


def _fit_sklearn(model: str, x_tr, y_tr, x_te, cats) -> np.ndarray:
    """Comparator models. Categoricals are ordinal-encoded from the training frame;
    unseen test categories map to -1 (a valid 'other' leaf for tree models)."""
    from sklearn.ensemble import (HistGradientBoostingRegressor,
                                   RandomForestRegressor)
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    # Rebuild frames as plain (non-categorical) columns; ordinal-encode categoricals
    # from the training vocabulary, unseen test categories -> -1.
    x_tr = x_tr.copy()
    x_te = x_te.copy()
    tr_cols, te_cols = {}, {}
    for c in x_tr.columns:
        if c in cats:
            cat = pd.Categorical(x_tr[c].astype("object"))
            mapping = {v: i for i, v in enumerate(cat.categories)}
            tr_cols[c] = pd.Series(cat.codes, index=x_tr.index).astype(int)
            te_cols[c] = x_te[c].astype("object").map(mapping).fillna(-1).astype(int)
        else:
            tr_cols[c] = pd.to_numeric(x_tr[c], errors="coerce")
            te_cols[c] = pd.to_numeric(x_te[c], errors="coerce")
    x_tr = pd.DataFrame(tr_cols, index=x_tr.index)
    x_te = pd.DataFrame(te_cols, index=x_te.index)

    if model == "ridge":
        est = Pipeline([("impute_scale", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
        x_tr = x_tr.fillna(x_tr.median())
        x_te = x_te.fillna(x_tr.median())
        est.fit(x_tr, y_tr)
        return est.predict(x_te)
    if model == "random_forest":
        est = RandomForestRegressor(n_estimators=300, n_jobs=-1,
                                    random_state=tpm.RANDOM_SEED)
        x_tr = x_tr.fillna(-1)
        x_te = x_te.fillna(-1)
        est.fit(x_tr, y_tr)
        return est.predict(x_te)
    if model == "hist_gbm":
        est = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05,
                                            random_state=tpm.RANDOM_SEED)
        est.fit(x_tr, y_tr)
        return est.predict(x_te)
    if model == "xgboost":
        import xgboost as xgb
        est = xgb.XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6,
                               subsample=0.9, colsample_bytree=0.9, n_jobs=-1,
                               random_state=tpm.RANDOM_SEED)
        est.fit(x_tr.fillna(np.nan), y_tr)
        return est.predict(x_te.fillna(np.nan))
    raise ValueError(f"unknown comparator model: {model}")


def fit_bundle(data: ExperimentData, features: Optional[List[str]] = None,
               *, monotone: bool = False) -> Dict:
    """Fit q10/q50/q90 (on proper-train) + mean model, mirroring production.

    Band models train on the proper-train slice so the calibration set is unseen;
    q50/mean train on the full train set. Returns log-space predictions on both the
    test and calibration sets for downstream conformal fitting. monotone=True applies
    the production monotone constraints.
    """
    features = features or data.feature_names
    xp, cats = _prep(data.x_ptrain, features, data.cat_cols)
    xc, _ = _prep(data.x_cal, features, data.cat_cols)
    xtr, _ = _prep(data.x_train, features, data.cat_cols)
    xte, _ = _prep(data.x_test, features, data.cat_cols)

    # monotone_constraints are only valid under the regression objective (the mean/
    # point model); LightGBM rejects them for quantile objectives, so the band models
    # stay unconstrained.
    mean_params = dict(LGBM_PARAMS)
    if monotone:
        mean_params["monotone_constraints"] = tpm.monotone_constraints_for(xtr.columns.tolist())

    def _fit(x, y, objective, params, **kw):
        m = tpm.make_lgbm(objective, params=params, **kw)
        m.fit(x, y, categorical_feature=cats if cats else "auto")
        return m

    q10 = _fit(xp, data.y_ptrain_log, "quantile", LGBM_PARAMS, alpha=0.10)
    q90 = _fit(xp, data.y_ptrain_log, "quantile", LGBM_PARAMS, alpha=0.90)
    q50 = _fit(xtr, data.y_train_log, "quantile", LGBM_PARAMS, alpha=0.50)
    mean = _fit(xtr, data.y_train_log, "regression", mean_params)

    return {
        "test": {
            "q10_log": q10.predict(xte), "q90_log": q90.predict(xte),
            "q50_log": q50.predict(xte), "point": np.expm1(mean.predict(xte)),
        },
        "cal": {
            "q10_log": q10.predict(xc), "q90_log": q90.predict(xc),
            "q50_log": q50.predict(xc),
        },
    }
