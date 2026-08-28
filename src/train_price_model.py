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

Why quantile regression: predicting q10/q50/q90 gives you a *range* a car
should sell for. A listing above q90 is overpriced; below q10 is a deal.
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


DATA_PATH = str(paths.PROCESSED_CSV)
TARGET_COLUMN = "price"

# Cols that aren't features (ids, leakage, raw helpers)
DROP_COLUMNS = [
    "listing_id", "listing_key", "zip",
    "price_per_mile",   # leaks the target - price/mileage
    "make", "model",    # redundant with make_model; let's not triple-count
]

# Categoricals - LightGBM handles them natively, no encoding needed
CATEGORICAL_FEATURES = [
    "make_model", "trim", "body_style", "mileage_bucket",
    "drivetrain", "fuel_type_group", "transmission_type", "car_color",
    "title_status", "accident_history", "region",
]

TEST_SIZE = 0.20
RANDOM_SEED = 42
CV_FOLDS = 5

# Conformal band calibration (CQR): fraction of the TRAIN split held out to calibrate
# the q10/q90 interval, and the coverage the calibrated band targets.
CALIBRATION_FRACTION = 0.20
TARGET_COVERAGE = 0.80
# Bins for Mondrian (group-conditional) conformal calibration; 10 is the operating
# point from the bin-count sweep (docs/paper_results.md): it roughly halves the
# conditional-coverage error vs marginal CQR for ~4% extra band width.
MONDRIAN_BINS = 10
# Enforce economic monotonicity (mileage/age down, year/tier/price-line up) on the
# production models. See MONOTONE_DIRECTIONS and docs/paper_results.md §monotonicity.
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
    drop_existing = [c for c in DROP_COLUMNS if c in cleaned.columns]

    x = cleaned.drop(columns=[TARGET_COLUMN] + drop_existing, errors="ignore")
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
    """CV in log-space, but report MAE in $ space for interpretability."""
    cv = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_SEED)
    maes = []
    r2s = []
    for fold_i, (tr, va) in enumerate(cv.split(x)):
        x_tr, x_va = x.iloc[tr], x.iloc[va]
        y_tr, y_va = y_log.iloc[tr], y_log.iloc[va]
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


# ---------------------------- main ----------------------------

def main() -> None:
    log("Loading dataset...")
    df = load_dataset(DATA_PATH)
    x, y, feature_names, cat_cols = prepare_features(df)
    log(f"Rows: {len(df)} -> usable: {len(x)} | features: {x.shape[1]} | categorical: {len(cat_cols)}")

    y_log = np.log1p(y)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_SEED,
    )
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)

    # Hold out a calibration slice from train for conformal band calibration (CQR).
    # Done up front because the make_model encoding is computed from the proper-train
    # slice only - the calibration and test sets must not encode their own prices.
    x_ptrain, x_cal, y_ptrain, y_cal = train_test_split(
        x_train, y_train, test_size=CALIBRATION_FRACTION, random_state=RANDOM_SEED,
    )
    y_ptrain_log = np.log1p(y_ptrain)
    y_cal_log = np.log1p(y_cal)

    mm_enc, mm_prior = compute_make_model_encoding(x_ptrain, y_ptrain_log)
    x_train = apply_make_model_encoding(x_train, mm_enc, mm_prior)
    x_test = apply_make_model_encoding(x_test, mm_enc, mm_prior)
    x_ptrain = apply_make_model_encoding(x_ptrain, mm_enc, mm_prior)
    x_cal = apply_make_model_encoding(x_cal, mm_enc, mm_prior)
    x = apply_make_model_encoding(x, mm_enc, mm_prior)  # for category_levels below
    feature_names = feature_names + [ENCODING_FEATURE]
    log(f"make_model target encoding: {len(mm_enc)} categories, prior={mm_prior:.3f}")

    base = baseline_metrics(y_train, y_test)
    log(f"Baseline -> MAE=${base['baseline_mae']:,.0f}, R2={base['baseline_r2']:.4f}")

    log("\n=== Tiny hyperparameter search (q50 model) ===")
    best_params, hp_results = small_hp_search(x_train, y_train_log, cat_cols)
    log(f"Best params: {best_params}")

    log("\n=== Final 5-fold CV with best params ===")
    cv = cv_score(lambda: make_lgbm("regression", params=best_params), x_train, y_train_log, cat_cols, folds=CV_FOLDS)
    log(f"CV MAE=${cv['cv_mae_mean']:,.0f} (+/- {cv['cv_mae_std']:,.0f}), R2={cv['cv_r2_mean']:.4f}")

    log("\n=== Training final models (q10, q50, q90) ===")
    models: Dict[str, lgb.LGBMRegressor] = {}

    # Economic monotonicity applies to the POINT model only: LightGBM forbids
    # monotone_constraints under the quantile objective, and the headline predicted
    # price (which the violation audit is about) is the mean model anyway. The band
    # quantiles stay unconstrained.
    mean_params = dict(best_params)
    if USE_MONOTONE_CONSTRAINTS:
        mean_params["monotone_constraints"] = monotone_constraints_for(x_train.columns.tolist())
        log(f"  monotone constraints (mean/point model): "
            f"{[(c, d) for c, d in zip(x_train.columns, mean_params['monotone_constraints']) if d]}")

    # Band models (q10, q90) train on the proper-train slice, leaving x_cal unseen.
    for name, alpha in [("q10", 0.10), ("q90", 0.90)]:
        log(f"  fitting {name} (on proper-train slice)...")
        m = make_lgbm("quantile", alpha=alpha, params=best_params)
        m.fit(
            x_ptrain, y_ptrain_log,
            categorical_feature=cat_cols if cat_cols else "auto",
            eval_set=[(x_test, y_test_log)],
            callbacks=[lgb.early_stopping(80, verbose=False)],
        )
        models[name] = m

    # q50 (display median) and the mean point model train on the FULL train set so
    # the headline point metrics don't pay for the calibration hold-out.
    for name, objective, kw in [("q50", "quantile", {"alpha": 0.50}), ("mean", "regression", {})]:
        label = "q50" if name == "q50" else "mean (regression L2) point"
        log(f"  fitting {label} model (on full train)...")
        params = mean_params if name == "mean" else best_params
        m = make_lgbm(objective, params=params, **kw)
        m.fit(
            x_train, y_train_log,
            categorical_feature=cat_cols if cat_cols else "auto",
            eval_set=[(x_test, y_test_log)],
            callbacks=[lgb.early_stopping(80, verbose=False)],
        )
        models[name] = m

    # Conformal calibration: measure how far actuals fall outside the raw band on the
    # untouched calibration slice, and derive the additive log-space adjustment.
    log("  calibrating conformal band (CQR)...")
    cal_lo_log = models["q10"].predict(x_cal)
    cal_hi_log = models["q90"].predict(x_cal)
    cal_mid_log = models["q50"].predict(x_cal)
    log_delta = conformal.compute_log_delta(
        cal_lo_log, cal_hi_log, y_cal_log, target_coverage=TARGET_COVERAGE,
    )
    conformal.save(ARTIFACT_DIR, log_delta, TARGET_COVERAGE, len(x_cal))
    log(f"  conformal log-delta={log_delta:+.4f} (target coverage {TARGET_COVERAGE:.0%}, "
        f"calibrated on {len(x_cal)} held-out rows)")

    # Mondrian calibration: separate deltas per predicted-price bin. This reduced
    # empirical coverage imbalance across price segments in the calibration study,
    # but it does not guarantee coverage for every segment. The marginal delta
    # above remains a backward-compatible fallback.
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

    # Apply the conformal adjustment to widen (or tighten) the band in log space.
    pred_q10 = np.expm1(pred_log_q10 - log_delta)
    pred_q50 = np.expm1(pred_log_q50)
    pred_q90 = np.expm1(pred_log_q90 + log_delta)
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

    # make_model target encoding - inference derives the same feature from this.
    (artifact_dir / "make_model_price_enc.json").write_text(json.dumps({
        "encoding": mm_enc, "prior": mm_prior, "smoothing": ENCODING_SMOOTHING,
    }, indent=2))

    # Category vocabularies: the exact levels the model saw for each categorical
    # feature. The predict-time validator uses these levels to
    # identify values outside the trained vocabulary, and
    # (b) flag unseen make_model / trim at inference time as low-confidence.
    category_levels = {
        c: [lvl for lvl in x[c].cat.categories.tolist()]
        for c in cat_cols if c in x.columns
    }
    (artifact_dir / "category_levels.json").write_text(json.dumps(category_levels, indent=2))

    # Category counts: how many training rows back each make_model / trim. Prediction
    # uses these to flag sparsely-observed values (e.g. a make_model with a handful of
    # listings) as low-confidence, not just entirely-unseen ones.
    HIGH_IMPACT_CATEGORICALS = ("make_model", "trim")
    category_counts = {
        c: {str(k): int(v) for k, v in x_train[c].value_counts().items()}
        for c in HIGH_IMPACT_CATEGORICALS if c in x_train.columns
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
            "data_path": DATA_PATH,
            "target": TARGET_COLUMN,
            "drop_columns": DROP_COLUMNS,
            "cat_features": cat_cols,
            "test_size": TEST_SIZE,
            "random_seed": RANDOM_SEED,
            "cv_folds": CV_FOLDS,
            "best_hyperparameters": best_params,
            "log_transform_target": True,
            "model": "lightgbm + quantile regression (q10/q50/q90) + mean point model",
        },
        "dataset": {
            "rows_total": int(len(df)),
            "rows_train": int(len(x_train)),
            "rows_test": int(len(x_test)),
            "features_used": int(x.shape[1]),
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

    # Version this model: snapshot the just-written artifacts into archive/vN/ and
    # record them in the registry so a future retrain never clobbers this one.
    import model_registry
    rows = payload["dataset"]["rows_total"]
    version = model_registry.register_current_model(
        artifact_dir, metrics_payload=payload,
        notes=f"trained on {rows} rows",
    )
    log(f"Registered model as v{version} (archived under {artifact_dir}/archive/v{version}/)")

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
