"""
Train a car price model with LightGBM, on log(price), with quantile regression
for price *bands* instead of a single point estimate.

Outputs (in model_artifacts/):
  - lgbm_q10.pkl, lgbm_q50.pkl, lgbm_q90.pkl     -- the three quantile models
  - feature_names.json                            -- column order for inference
  - cat_features.json                             -- which cols are categorical
  - metrics.json                                  -- baseline + test metrics + CV
  - test_predictions.csv                          -- actual / pred / lo / hi / verdict
  - feature_importance.csv                        -- gain-based importance from q50
  - quick_model_writeup.md                        -- human-readable summary

Why quantile regression: predicting q10/q50/q90 gives a range of asking prices
seen for comparable vehicles. A listing above q90 is high relative to that range;
one below q10 is low.
"""

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

import lightgbm as lgb

import paths
import conformal
import conformal_mondrian


DATA_PATH = paths.PROCESSED_CSV
TARGET_COLUMN = "price"

# Cols that aren't features (ids, leakage, raw helpers)
DROP_COLUMNS = [
    "listing_id", "listing_key", "zip",
    "price_per_mile",   # leaks the target - price/mileage
    "make", "model",    # redundant with make_model; let's not triple-count
    "region",           # sparse geographic tag on a minority of rows; not a
                        # modeled signal - kept in the data as metadata only
]

# The training contract is explicit so an extra column in a future processed file
# cannot silently become a model input (especially a target-derived field).
MODEL_FEATURES = [
    "make_model", "year", "mileage", "car_age", "miles_per_year",
    "mileage_bucket", "is_luxury_make", "is_new", "trim", "trim_tier",
    "title_status", "accident_history", "owner_count", "cpo", "drivetrain",
    "engine_size_l", "transmission_type", "fuel_type_group", "mpg",
    "car_color", "body_style",
]

# Categoricals - LightGBM handles them natively, no encoding needed
CATEGORICAL_FEATURES = [
    "make_model", "trim", "body_style", "mileage_bucket",
    "drivetrain", "fuel_type_group", "transmission_type", "car_color",
    "title_status", "accident_history",
]

TEST_SIZE = 0.20
RANDOM_SEED = 42
CV_FOLDS = 5
EARLY_STOP_VALIDATION_FRACTION = 0.15
EARLY_STOPPING_ROUNDS = 80

# Conformal band calibration (CQR): fraction of the TRAIN split held out to calibrate
# the q10/q90 interval, and the coverage the calibrated band targets.
CALIBRATION_FRACTION = 0.20
TARGET_COVERAGE = 0.80
# Bins for Mondrian (group-conditional) conformal calibration; 10 is the operating
# point selected in the saved calibration experiments: it roughly halves the
# conditional-coverage error versus marginal CQR for about 4% extra band width.
MONDRIAN_BINS = 10
# Enforce economic monotonicity (mileage/age down, year/tier/price-line up) on the
# production models. See MONOTONE_DIRECTIONS below.
USE_MONOTONE_CONSTRAINTS = True

ARTIFACT_DIR = str(paths.ARTIFACT_DIR)


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------- data ----------------------------

def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in {path}")
    return df


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    cleaned = df.copy().dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    missing = [column for column in MODEL_FEATURES if column not in cleaned.columns]
    if missing:
        raise ValueError(f"Processed data is missing required model features: {missing}")

    x = cleaned[MODEL_FEATURES].copy()
    y = pd.to_numeric(cleaned[TARGET_COLUMN], errors="coerce")

    valid = y.notna() & (y > 0)
    x = x.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)

    # cast categoricals as pandas 'category' so LightGBM treats them right
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in x.columns]
    for c in cat_cols:
        x[c] = x[c].astype("category")

    # numerics: coerce
    for c in x.columns:
        if c not in cat_cols:
            x[c] = pd.to_numeric(x[c], errors="coerce")

    # drop columns that are 100% NaN - they help no one
    all_nan = [c for c in x.columns if x[c].isna().all()]
    if all_nan:
        log(f"Dropping all-NaN columns: {all_nan}")
        x = x.drop(columns=all_nan)
        cat_cols = [c for c in cat_cols if c in x.columns]

    return x, y, x.columns.tolist(), cat_cols


def infer_reference_year(x: pd.DataFrame) -> int:
    """Recover the year used to derive car_age in the processed snapshot."""
    candidates = (
        pd.to_numeric(x["year"], errors="coerce")
        + pd.to_numeric(x["car_age"], errors="coerce")
    ).dropna().round().astype(int)
    plausible = candidates[candidates.between(2000, 2100)]
    if plausible.empty:
        raise ValueError("Could not infer the preprocessing reference year.")
    return int(plausible.mode().iloc[0])


# ---------------------------- modeling ----------------------------

# Economically-required monotonicity, enforced as LightGBM monotone_constraints.
# Directions are in log1p(price) space, but monotonicity is preserved by expm1, so
# they hold in dollars too. Constraining these (a) removes nonsensical predictions
# (an unconstrained model raises the price of ~1 in 4 cars when you add mileage),
# and (b) regularizes extrapolation into the sparse luxury region, where the
# make_model price encoding and trim tier carry most of the signal.
# The "depreciation" constraint set: the three unarguable physical-depreciation
# directions. Empirically these drive ceteris-paribus monotonicity violations to
# exactly zero at no accuracy cost (MAE unchanged within noise), whereas also
# constraining the collinear derived features (car_age, miles_per_year) or the
# learned signals (make_model_price_enc, trim_tier) over-regularizes and measurably
# hurts accuracy - a legitimately loud luxury car in poor condition *should* be
# allowed to price below its model line's typical value.
MONOTONE_DIRECTIONS = {
    "year": +1,                    # newer is not cheaper, all else equal
    "mileage": -1,                 # more miles is not more expensive
    "owner_count": -1,             # more prior owners is not more expensive
}


def monotone_constraints_for(columns) -> list:
    """Build LightGBM's monotone_constraints vector for a given feature order.
    Categoricals and unconstrained numerics get 0."""
    return [MONOTONE_DIRECTIONS.get(c, 0) for c in columns]


def make_lgbm(objective: str, alpha: float = None, params: Dict = None) -> lgb.LGBMRegressor:
    base = dict(
        n_estimators=2000,
        learning_rate=0.04,
        num_leaves=127,
        min_child_samples=20,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=1,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )
    if params:
        base.update(params)
    base["objective"] = objective
    if objective == "quantile":
        base["alpha"] = alpha
    return lgb.LGBMRegressor(**base)


# --- make_model target encoding ---
# With hundreds of make_model categories and min_child_samples floors, LightGBM
# can't isolate a 15-listing model line (an AMG G-Wagon, a 911) into its own leaf,
# so rare-but-expensive lines get shrunk toward the market average and luxury cars
# come out badly underpredicted. A smoothed mean-log-price-per-make_model feature
# turns that into a single continuous split. Computed from the proper-train slice
# only, so the calibration and test sets never contribute to their own encoding.
ENCODING_FEATURE = "make_model_price_enc"
ENCODING_SMOOTHING = 10.0


def compute_make_model_encoding(x: pd.DataFrame, y_log: pd.Series) -> Tuple[Dict[str, float], float]:
    prior = float(y_log.mean())
    grouped = y_log.groupby(x["make_model"].astype(str), observed=True)
    smoothed = (grouped.sum() + prior * ENCODING_SMOOTHING) / (grouped.count() + ENCODING_SMOOTHING)
    return {str(k): float(v) for k, v in smoothed.items()}, prior


def apply_make_model_encoding(x: pd.DataFrame, enc: Dict[str, float], prior: float) -> pd.DataFrame:
    x = x.copy()
    x[ENCODING_FEATURE] = x["make_model"].astype(str).map(enc).fillna(prior).astype(float)
    return x


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(y_true), 1e-9)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def baseline_metrics(y_train: pd.Series, y_test: pd.Series) -> Dict[str, float]:
    median_pred = np.full(len(y_test), float(np.median(y_train)))
    return {
        "baseline_mae": float(mean_absolute_error(y_test, median_pred)),
        "baseline_rmse": float(np.sqrt(mean_squared_error(y_test, median_pred))),
        "baseline_r2": float(r2_score(y_test, median_pred)),
        "baseline_mape_percent": safe_mape(y_test.to_numpy(), median_pred),
    }


def cv_score(model_fn, x: pd.DataFrame, y_log: pd.Series, cat_cols: List[str], folds: int = CV_FOLDS) -> Dict[str, float]:
    """CV in log-space, with target encoding fitted inside each training fold."""
    cv = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_SEED)
    maes = []
    r2s = []
    for fold_i, (tr, va) in enumerate(cv.split(x)):
        x_tr_raw, x_va_raw = x.iloc[tr], x.iloc[va]
        y_tr, y_va = y_log.iloc[tr], y_log.iloc[va]
        fold_enc, fold_prior = compute_make_model_encoding(x_tr_raw, y_tr)
        x_tr = apply_make_model_encoding(x_tr_raw, fold_enc, fold_prior)
        x_va = apply_make_model_encoding(x_va_raw, fold_enc, fold_prior)
        model = model_fn()
        model.fit(
            x_tr, y_tr,
            categorical_feature=cat_cols if cat_cols else "auto",
            eval_set=[(x_va, y_va)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        pred_log = model.predict(x_va)
        pred_dollars = np.expm1(pred_log)
        actual_dollars = np.expm1(y_va)
        maes.append(mean_absolute_error(actual_dollars, pred_dollars))
        r2s.append(r2_score(actual_dollars, pred_dollars))
    return {
        "cv_mae_mean": float(np.mean(maes)),
        "cv_mae_std": float(np.std(maes)),
        "cv_r2_mean": float(np.mean(r2s)),
    }


def small_hp_search(x: pd.DataFrame, y_log: pd.Series, cat_cols: List[str]) -> Tuple[Dict, Dict]:
    """Tiny grid - enough to noticeably help, fast enough to run on a laptop."""
    grid = [
        dict(num_leaves=63,  learning_rate=0.05, min_child_samples=20),
        dict(num_leaves=127, learning_rate=0.04, min_child_samples=20),
        dict(num_leaves=255, learning_rate=0.03, min_child_samples=30),
        dict(num_leaves=127, learning_rate=0.03, min_child_samples=10),
    ]
    results = {}
    best = (None, float("inf"))
    for i, params in enumerate(grid):
        log(f"  HP combo {i+1}/{len(grid)}: {params}")
        scores = cv_score(lambda p=params: make_lgbm("regression", params=p), x, y_log, cat_cols, folds=3)
        results[json.dumps(params)] = scores
        log(f"    CV MAE=${scores['cv_mae_mean']:,.0f} (+/- {scores['cv_mae_std']:,.0f}), R2={scores['cv_r2_mean']:.4f}")
        if scores["cv_mae_mean"] < best[1]:
            best = (params, scores["cv_mae_mean"])
    return best[0], results


def fit_final_model(
    objective: str,
    x_full: pd.DataFrame,
    y_full: pd.Series,
    cat_cols: List[str],
    params: Dict,
    *,
    encoding: Dict[str, float],
    encoding_prior: float,
    alpha: float = None,
    seed_offset: int = 0,
) -> Tuple[lgb.LGBMRegressor, int]:
    """Select an iteration count internally, then refit on all allowed rows.

    The outer test split is never passed to LightGBM during fitting. Quantile
    models call this with the proper-training rows so the separate conformal
    calibration slice also remains untouched.
    """
    x_fit_raw, x_val_raw, y_fit, y_val = train_test_split(
        x_full,
        y_full,
        test_size=EARLY_STOP_VALIDATION_FRACTION,
        random_state=RANDOM_SEED + seed_offset,
    )
    probe_enc, probe_prior = compute_make_model_encoding(x_fit_raw, y_fit)
    x_fit = apply_make_model_encoding(x_fit_raw, probe_enc, probe_prior)
    x_val = apply_make_model_encoding(x_val_raw, probe_enc, probe_prior)
    probe = make_lgbm(objective, alpha=alpha, params=params)
    probe.fit(
        x_fit,
        y_fit,
        categorical_feature=cat_cols if cat_cols else "auto",
        eval_set=[(x_val, y_val)],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    best_iteration = int(probe.best_iteration_ or probe.n_estimators)

    final_params = dict(params)
    final_params["n_estimators"] = best_iteration
    model = make_lgbm(objective, alpha=alpha, params=final_params)
    x_final = apply_make_model_encoding(x_full, encoding, encoding_prior)
    model.fit(
        x_final,
        y_full,
        categorical_feature=cat_cols if cat_cols else "auto",
    )
    return model, best_iteration


# ---------------------------- main ----------------------------

def main() -> None:
    log("Loading dataset...")
    df = load_dataset(DATA_PATH)
    x, y, feature_names, cat_cols = prepare_features(df)
    reference_year = infer_reference_year(x)
    log(f"Rows: {len(df)} -> usable: {len(x)} | features: {x.shape[1]} | categorical: {len(cat_cols)}")
    log(f"Feature reference year: {reference_year}")

    y_log = np.log1p(y)

    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_SEED,
    )
    y_train_log = np.log1p(y_train)
    # Hold out a calibration slice from train for conformal band calibration (CQR).
    # Done up front because the make_model encoding is computed from the proper-train
    # slice only - the calibration and test sets must not encode their own prices.
    x_ptrain_raw, x_cal_raw, y_ptrain, y_cal = train_test_split(
        x_train_raw, y_train, test_size=CALIBRATION_FRACTION, random_state=RANDOM_SEED,
    )
    y_ptrain_log = np.log1p(y_ptrain)
    y_cal_log = np.log1p(y_cal)

    base = baseline_metrics(y_train, y_test)
    log(f"Baseline -> MAE=${base['baseline_mae']:,.0f}, R2={base['baseline_r2']:.4f}")

    # Model selection uses only the proper-training rows. Each CV fold fits its own
    # make/model encoding, so neither validation-fold prices nor calibration prices
    # enter the features used to score a candidate configuration.
    log("\n=== Tiny hyperparameter search (proper-training rows only) ===")
    best_params, hp_results = small_hp_search(x_ptrain_raw, y_ptrain_log, cat_cols)
    log(f"Best params: {best_params}")

    log("\n=== Final 5-fold CV with best params (proper-training rows only) ===")
    cv = cv_score(
        lambda: make_lgbm("regression", params=best_params),
        x_ptrain_raw,
        y_ptrain_log,
        cat_cols,
        folds=CV_FOLDS,
    )
    log(f"CV MAE=${cv['cv_mae_mean']:,.0f} (+/- {cv['cv_mae_std']:,.0f}), R2={cv['cv_r2_mean']:.4f}")

    # Fit the production encoding after model selection. Calibration and test rows
    # do not contribute to it.
    mm_enc, mm_prior = compute_make_model_encoding(x_ptrain_raw, y_ptrain_log)
    x_train = apply_make_model_encoding(x_train_raw, mm_enc, mm_prior)
    x_test = apply_make_model_encoding(x_test_raw, mm_enc, mm_prior)
    x_cal = apply_make_model_encoding(x_cal_raw, mm_enc, mm_prior)
    feature_names = feature_names + [ENCODING_FEATURE]
    log(f"make_model target encoding: {len(mm_enc)} categories, prior={mm_prior:.3f}")

    log("\n=== Training final models (q10, q50, q90, point) ===")
    models: Dict[str, lgb.LGBMRegressor] = {}
    selected_iterations: Dict[str, int] = {}

    # Economic monotonicity applies to the POINT model only: LightGBM forbids
    # monotone_constraints under the quantile objective, and the headline predicted
    # price (which the violation audit is about) is the mean model anyway. The band
    # quantiles stay unconstrained.
    mean_params = dict(best_params)
    if USE_MONOTONE_CONSTRAINTS:
        mean_params["monotone_constraints"] = monotone_constraints_for(x_train.columns.tolist())
        log(f"  monotone constraints (mean/point model): "
            f"{[(c, d) for c, d in zip(x_train.columns, mean_params['monotone_constraints']) if d]}")

    # All quantile models train on the proper-training slice, leaving calibration
    # and test rows unseen. Each chooses its iteration count on an internal split,
    # then refits on the full proper-training slice.
    for offset, (name, alpha) in enumerate([("q10", 0.10), ("q50", 0.50), ("q90", 0.90)], start=1):
        log(f"  fitting {name} (on proper-train slice)...")
        m, best_iteration = fit_final_model(
            "quantile",
            x_ptrain_raw, y_ptrain_log,
            cat_cols,
            best_params,
            encoding=mm_enc,
            encoding_prior=mm_prior,
            alpha=alpha,
            seed_offset=offset,
        )
        models[name] = m
        selected_iterations[name] = best_iteration
        log(f"    selected {best_iteration} boosting rounds")

    # The point model can use the full training split because it is not calibrated.
    # Its boosting rounds are still chosen without looking at the outer test split.
    log("  fitting mean (regression L2) point model (on full train)...")
    models["mean"], selected_iterations["mean"] = fit_final_model(
        "regression",
        x_train_raw,
        y_train_log,
        cat_cols,
        mean_params,
        encoding=mm_enc,
        encoding_prior=mm_prior,
        seed_offset=10,
    )
    log(f"    selected {selected_iterations['mean']} boosting rounds")

    # Conformal calibration: measure how far actuals fall outside the raw band on the
    # untouched calibration slice, and derive the additive log-space adjustment.
    log("  calibrating conformal band (CQR)...")
    cal_q10_log = models["q10"].predict(x_cal)
    cal_q90_log = models["q90"].predict(x_cal)
    cal_lo_log = np.minimum(cal_q10_log, cal_q90_log)
    cal_hi_log = np.maximum(cal_q10_log, cal_q90_log)
    cal_mid_log = models["q50"].predict(x_cal)
    log_delta = conformal.compute_log_delta(
        cal_lo_log, cal_hi_log, y_cal_log, target_coverage=TARGET_COVERAGE,
    )
    conformal.save(ARTIFACT_DIR, log_delta, TARGET_COVERAGE, len(x_cal))
    log(f"  conformal log-delta={log_delta:+.4f} (target coverage {TARGET_COVERAGE:.0%}, "
        f"calibrated on {len(x_cal)} held-out rows)")

    # Mondrian (group-conditional) calibration: separate deltas per predicted-price
    # bin. This improved empirical coverage balance across price segments in the
    # calibration study, but does not guarantee coverage for every segment. It is
    # served preferentially by predict_listing; the marginal delta above remains a
    # backward-compatible fallback.
    mondrian_model = conformal_mondrian.fit(
        cal_lo_log, cal_hi_log, cal_mid_log, y_cal_log,
        target_coverage=TARGET_COVERAGE, n_bins=MONDRIAN_BINS,
    )
    conformal_mondrian.save(ARTIFACT_DIR, mondrian_model)
    log(f"  mondrian deltas ({MONDRIAN_BINS} bins): "
        f"{[round(x, 3) for x in mondrian_model['bin_deltas_log']]}")

    # ------------------ evaluation in $ space ------------------
    pred_log_q10 = models["q10"].predict(x_test)
    pred_log_q50 = models["q50"].predict(x_test)
    pred_log_q90 = models["q90"].predict(x_test)
    pred_log_pt  = models["mean"].predict(x_test)

    # Apply the same price-segment-specific deltas used by predict_listing.py.
    lo_delta, hi_delta = conformal_mondrian.deltas_for(mondrian_model, pred_log_q50)
    band_lo_log, median_log, band_hi_log = conformal.calibrated_band_logs(
        pred_log_q10, pred_log_q50, pred_log_q90, lo_delta, hi_delta,
    )
    pred_q10 = np.expm1(band_lo_log)
    pred_q50 = np.expm1(median_log)
    pred_q90 = np.expm1(band_hi_log)
    pred_pt  = np.expm1(pred_log_pt)

    actual = y_test.to_numpy()

    test_metrics = {
        "test_mae":  float(mean_absolute_error(actual, pred_pt)),
        "test_rmse": float(np.sqrt(mean_squared_error(actual, pred_pt))),
        "test_r2":   float(r2_score(actual, pred_pt)),
        "test_mape_percent": safe_mape(actual, pred_pt),
        "median_q50_mae": float(mean_absolute_error(actual, pred_q50)),
        "interval_coverage_q10_q90": float(np.mean((actual >= pred_q10) & (actual <= pred_q90))),
        "interval_avg_width_dollars": float(np.mean(pred_q90 - pred_q10)),
    }
    log(
        f"\nTest -> MAE=${test_metrics['test_mae']:,.0f}, "
        f"RMSE=${test_metrics['test_rmse']:,.0f}, "
        f"R2={test_metrics['test_r2']:.4f}, "
        f"MAPE={test_metrics['test_mape_percent']:.2f}%"
    )
    log(
        f"q10-q90 interval covers {test_metrics['interval_coverage_q10_q90']*100:.1f}% of actuals "
        f"(target ~80%); avg width ${test_metrics['interval_avg_width_dollars']:,.0f}"
    )

    # verdict labels for sample predictions
    verdicts = []
    for a, lo, hi in zip(actual, pred_q10, pred_q90):
        if a > hi:
            verdicts.append("overpriced")
        elif a < lo:
            verdicts.append("deal")
        else:
            verdicts.append("fair")

    pred_df = pd.DataFrame({
        "actual_price": actual,
        "predicted_price": pred_pt.round(0),
        "lo_q10": pred_q10.round(0),
        "median_q50": pred_q50.round(0),
        "hi_q90": pred_q90.round(0),
        "abs_error": np.abs(actual - pred_pt).round(0),
        "verdict_if_listed_at_actual": verdicts,
    })

    # ---------------- save artifacts ----------------
    artifact_dir = Path(ARTIFACT_DIR)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    for name, m in models.items():
        with open(artifact_dir / f"lgbm_{name}.pkl", "wb") as f:
            pickle.dump(m, f)

    (artifact_dir / "feature_names.json").write_text(json.dumps(feature_names, indent=2))
    (artifact_dir / "cat_features.json").write_text(json.dumps(cat_cols, indent=2))
    (artifact_dir / "model_metadata.json").write_text(json.dumps({
        "reference_year": reference_year,
        "target": "advertised asking price",
    }, indent=2))

    # make_model target encoding - inference derives the same feature from this.
    (artifact_dir / "make_model_price_enc.json").write_text(json.dumps({
        "encoding": mm_enc, "prior": mm_prior, "smoothing": ENCODING_SMOOTHING,
    }, indent=2))

    # Category vocabularies: the exact levels the model saw for each categorical
    # feature. The prediction-time validator uses these levels to flag unseen
    # make_model or trim values as low-confidence.
    category_levels = {
        c: sorted(str(value) for value in x_ptrain_raw[c].dropna().astype(str).unique())
        for c in cat_cols if c in x_ptrain_raw.columns
    }
    (artifact_dir / "category_levels.json").write_text(json.dumps(category_levels, indent=2))

    # Category counts: how many training rows back each make_model / trim. Prediction
    # uses these to flag sparsely-observed values (e.g. a make_model with a handful of
    # listings) as low-confidence, not just entirely-unseen ones.
    HIGH_IMPACT_CATEGORICALS = ("make_model", "trim")
    category_counts = {
        c: {str(k): int(v) for k, v in x_ptrain_raw[c].value_counts().items()}
        for c in HIGH_IMPACT_CATEGORICALS if c in x_ptrain_raw.columns
    }
    (artifact_dir / "category_counts.json").write_text(json.dumps(category_counts, indent=2))

    # feature importance from q50 (median model)
    fi = pd.DataFrame({
        "feature": feature_names,
        "gain_importance": models["q50"].booster_.feature_importance(importance_type="gain"),
        "split_importance": models["q50"].booster_.feature_importance(importance_type="split"),
    }).sort_values("gain_importance", ascending=False)
    fi.to_csv(artifact_dir / "feature_importance.csv", index=False)

    pred_df.to_csv(artifact_dir / "test_predictions.csv", index=False)

    payload = {
        "config": {
            "data_path": str(Path(DATA_PATH).relative_to(paths.PROJECT_ROOT)),
            "target": TARGET_COLUMN,
            "drop_columns": DROP_COLUMNS,
            "cat_features": cat_cols,
            "test_size": TEST_SIZE,
            "random_seed": RANDOM_SEED,
            "cv_folds": CV_FOLDS,
            "early_stop_validation_fraction": EARLY_STOP_VALIDATION_FRACTION,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "selected_iterations": selected_iterations,
            "reference_year": reference_year,
            "best_hyperparameters": best_params,
            "log_transform_target": True,
            "model": "lightgbm + quantile regression (q10/q50/q90) + mean point model",
        },
        "dataset": {
            "rows_total": int(len(df)),
            "rows_train": int(len(x_train)),
            "rows_calibration": int(len(x_cal)),
            "rows_test": int(len(x_test)),
            "features_used": int(len(feature_names)),
            "categorical_features": int(len(cat_cols)),
        },
        "baseline": base,
        "hp_search_results": hp_results,
        "cv_with_best_params": cv,
        "test_metrics": test_metrics,
    }
    (artifact_dir / "metrics.json").write_text(json.dumps(payload, indent=2))

    # quick writeup
    write_quick_writeup(payload, pred_df, fi, artifact_dir / "quick_model_writeup.md")

    # Record the run and keep a local snapshot of the just-written artifacts.
    import model_registry
    rows = payload["dataset"]["rows_total"]
    version = model_registry.register_current_model(
        artifact_dir, metrics_payload=payload,
        notes=(
            f"retrained on {rows:,} rows with fold-local target encoding and internal "
            "early-stopping validation; calibration and outer test rows remained "
            "untouched during model selection"
        ),
    )
    log(f"Registered model as v{version} (local snapshot: {artifact_dir}/archive/v{version}/)")

    log(f"\nSaved artifacts to {artifact_dir}/")


def write_quick_writeup(payload: Dict, pred_df: pd.DataFrame, fi: pd.DataFrame, path: Path) -> None:
    tm = payload["test_metrics"]
    base = payload["baseline"]
    fmt = lambda v: f"${v:,.0f}"

    top_features = fi.head(10).to_string(index=False)

    if not pred_df.empty:
        best = pred_df.nsmallest(1, "abs_error").iloc[0]
        worst = pred_df.nlargest(1, "abs_error").iloc[0]
        deals = pred_df[pred_df["verdict_if_listed_at_actual"] == "deal"].head(3)
        overs = pred_df[pred_df["verdict_if_listed_at_actual"] == "overpriced"].head(3)

    lines = [
        "# Quick Model Writeup",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Setup",
        f"- Model: {payload['config']['model']}",
        f"- Rows: total {payload['dataset']['rows_total']}, train {payload['dataset']['rows_train']}, test {payload['dataset']['rows_test']}",
        f"- Features: {payload['dataset']['features_used']} ({payload['dataset']['categorical_features']} categorical)",
        "",
        "## Headline results",
        "- Evaluation note: this fixed split was consulted during model development; see README.md.",
        f"- Test R2: {tm['test_r2']:.4f}  (baseline R2: {base['baseline_r2']:.4f})",
        f"- Test MAE: {fmt(tm['test_mae'])}  (baseline MAE: {fmt(base['baseline_mae'])})",
        f"- Test RMSE: {fmt(tm['test_rmse'])}",
        f"- Test MAPE: {tm['test_mape_percent']:.2f}%",
        "",
        "## Confidence band (q10-q90)",
        f"- Coverage on test set: {tm['interval_coverage_q10_q90']*100:.1f}% (ideal ~80%)",
        f"- Average band width: {fmt(tm['interval_avg_width_dollars'])}",
        "",
        "## Top features (gain)",
        "```",
        top_features,
        "```",
    ]

    if not pred_df.empty:
        lines += [
            "",
            "## Sample predictions",
            f"- Closest call: predicted {fmt(best['predicted_price'])}, actual {fmt(best['actual_price'])}, error {fmt(best['abs_error'])}",
            f"- Worst miss:  predicted {fmt(worst['predicted_price'])}, actual {fmt(worst['actual_price'])}, error {fmt(worst['abs_error'])}",
        ]
        if not deals.empty:
            lines += ["", "### Listings that look like deals (actual < q10):"]
            for _, r in deals.iterrows():
                lines.append(f"- listed {fmt(r['actual_price'])}, fair range {fmt(r['lo_q10'])}-{fmt(r['hi_q90'])}")
        if not overs.empty:
            lines += ["", "### Listings that look overpriced (actual > q90):"]
            for _, r in overs.iterrows():
                lines.append(f"- listed {fmt(r['actual_price'])}, fair range {fmt(r['lo_q10'])}-{fmt(r['hi_q90'])}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
