"""
Evaluation of the trained vehicle price model.

Where train_price_model.py reports a single headline MAE/R2, this script adds:

  1. Baselines    - does the model actually beat dumb-but-reasonable strategies
                    (global median, same-make/model median, a linear depreciation fit)?
  2. Where it fails - segment-level error analysis by price band, car age, mileage, and
                    luxury vs economy. A single average hides the luxury blind spot.
  3. Is the band honest - calibration of the q10/q50/q90 quantiles. If we claim an 80%
                    band, do ~80% of real prices actually fall inside it?
  4. Why            - SHAP global importances and a reusable per-listing explanation
                    function.

All numbers are computed on the same fixed evaluation split as training (same seed), so
they are directly comparable to metrics.json. Those rows were not used to fit or
calibrate version 13, but the split was consulted across several model versions. The
results are therefore model-development measurements, not a one-time final test.

IMPORTANT framing: the target is *asking* price from listings, not realized sale price.
Every "deal / fair / overpriced" statement is market-relative - priced low/in-line/high
versus comparable listings - not a claim about a car's true resale value.

Run:  python3 evaluate_model.py
Outputs land in model_artifacts/eval/.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Reuse the exact feature pipeline + split config from training so evaluation is
# apples-to-apples with metrics.json (no duplicated, drift-prone logic).
from train_price_model import (
    DATA_PATH,
    RANDOM_SEED,
    TEST_SIZE,
    load_dataset,
    prepare_features,
)
import paths
import conformal
import conformal_mondrian

ARTIFACT_DIR = paths.ARTIFACT_DIR
EVAL_DIR = paths.EVAL_DIR

QUANTILE_MODELS = {"q10": "lgbm_q10.pkl", "q50": "lgbm_q50.pkl", "q90": "lgbm_q90.pkl"}
POINT_MODEL = "lgbm_mean.pkl"


def log(msg: str) -> None:
    print(msg, flush=True)


# ----------------------------- metrics -----------------------------

def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(y_true), 1e-9)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def metric_block(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape_percent": safe_mape(y_true, y_pred),
        "n": int(len(y_true)),
    }


# ----------------------------- loading -----------------------------

def load_models() -> dict:
    models = {}
    for name, fname in QUANTILE_MODELS.items():
        with open(ARTIFACT_DIR / fname, "rb") as f:
            models[name] = pickle.load(f)
    with open(ARTIFACT_DIR / POINT_MODEL, "rb") as f:
        models["mean"] = pickle.load(f)
    return models


def predict_dollars(model, x: pd.DataFrame) -> np.ndarray:
    """Models are trained on log1p(price); invert to dollars."""
    return np.expm1(model.predict(x))


# ----------------------------- baselines -----------------------------

def baseline_global_median(y_train: pd.Series, n_test: int) -> np.ndarray:
    return np.full(n_test, float(np.median(y_train)))


def baseline_group_median(
    train_df: pd.DataFrame, test_df: pd.DataFrame, y_train: pd.Series
) -> np.ndarray:
    """Predict the median asking price of the same make_model seen in training.

    Falls back to the global median for unseen make_models. This represents a
    simple comparable-vehicle lookup.
    """
    global_median = float(np.median(y_train))
    grp = (
        pd.DataFrame({"make_model": train_df["make_model"].values, "price": y_train.values})
        .groupby("make_model")["price"]
        .median()
    )
    return np.array([grp.get(mm, global_median) for mm in test_df["make_model"].values])


def baseline_linear_depreciation(
    x_train: pd.DataFrame, y_train_log: pd.Series, x_test: pd.DataFrame
) -> np.ndarray:
    """A transparent linear model on the classic depreciation drivers:
    age, mileage, and a luxury flag. Predicts in log space, like the real model."""
    cols = [c for c in ["car_age", "mileage", "is_luxury_make"] if c in x_train.columns]
    xtr = x_train[cols].apply(pd.to_numeric, errors="coerce")
    xte = x_test[cols].apply(pd.to_numeric, errors="coerce")
    fill = xtr.median()
    xtr = xtr.fillna(fill)
    xte = xte.fillna(fill)
    lr = LinearRegression().fit(xtr, y_train_log)
    return np.expm1(lr.predict(xte))


# ----------------------------- segment analysis -----------------------------

def segment_errors(
    test_df: pd.DataFrame, actual: np.ndarray, pred_pt: np.ndarray,
    pred_lo: np.ndarray, pred_hi: np.ndarray,
) -> pd.DataFrame:
    """MAE / MAPE / band-coverage broken out by interpretable segments."""
    df = test_df.copy()
    df["_actual"] = actual
    df["_pred"] = pred_pt
    df["_abs_err"] = np.abs(actual - pred_pt)
    df["_ape"] = np.abs((actual - pred_pt) / np.maximum(actual, 1e-9)) * 100
    df["_in_band"] = (actual >= pred_lo) & (actual <= pred_hi)

    df["price_band"] = pd.cut(
        df["_actual"], bins=[0, 15000, 30000, 50000, 80000, np.inf],
        labels=["<15k", "15-30k", "30-50k", "50-80k", "80k+"],
    )
    if "car_age" in df.columns:
        df["age_band"] = pd.cut(
            pd.to_numeric(df["car_age"], errors="coerce"),
            bins=[-1, 2, 5, 10, np.inf], labels=["0-2y", "3-5y", "6-10y", "10y+"],
        )
    df["luxury"] = np.where(
        pd.to_numeric(df.get("is_luxury_make", 0), errors="coerce").fillna(0) == 1,
        "luxury", "economy",
    )

    rows: List[dict] = []

    def summarize(group_col: str) -> None:
        if group_col not in df.columns:
            return
        for value, g in df.groupby(group_col, observed=True):
            if len(g) == 0:
                continue
            rows.append({
                "segment_type": group_col,
                "segment": str(value),
                "n": int(len(g)),
                "mae": round(float(g["_abs_err"].mean()), 0),
                "mape_percent": round(float(g["_ape"].mean()), 1),
                "band_coverage": round(float(g["_in_band"].mean()), 3),
            })

    for col in ["price_band", "age_band", "mileage_bucket", "luxury"]:
        summarize(col)

    return pd.DataFrame(rows)


# ----------------------------- calibration -----------------------------

def quantile_calibration(
    actual: np.ndarray, q_preds: Dict[str, np.ndarray]
) -> pd.DataFrame:
    """Check raw quantile models before conformal interval adjustment."""
    targets = {"q10": 0.10, "q50": 0.50, "q90": 0.90}
    rows = []
    for name, level in targets.items():
        empirical = float(np.mean(actual <= q_preds[name]))
        rows.append({"quantile": name, "target_level": level,
                     "empirical_below": round(empirical, 3)})
    return pd.DataFrame(rows)


def plot_calibration(calib_df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    ax.plot(calib_df["target_level"], calib_df["empirical_below"], "o-",
            color="#2563eb", label="model")
    for _, r in calib_df.iterrows():
        ax.annotate(r["quantile"], (r["target_level"], r["empirical_below"]),
                    textcoords="offset points", xytext=(6, -10), fontsize=9)
    ax.set_xlabel("target quantile level")
    ax.set_ylabel("empirical fraction of actuals below prediction")
    ax.set_title("Raw quantile calibration")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ----------------------------- SHAP -----------------------------

def shap_global(models: dict, x_test: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    import shap
    explainer = shap.TreeExplainer(models["mean"])
    sv = explainer.shap_values(x_test)
    mean_abs = np.abs(sv).mean(axis=0)
    fi = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    top = fi.head(12).iloc[::-1]
    ax.barh(top["feature"], top["mean_abs_shap"], color="#2563eb")
    ax.set_xlabel("mean |SHAP| (impact on log-price prediction)")
    ax.set_title("Global feature importance (SHAP)")
    fig.tight_layout()
    fig.savefig(EVAL_DIR / "shap_global.png", dpi=120)
    plt.close(fig)
    return fi


def make_listing_explainer(models: dict, feature_names: List[str], cat_features: List[str]):
    """Return the top signed feature contributions for one listing."""
    import shap
    explainer = shap.TreeExplainer(models["mean"])

    def explain(x_row: pd.DataFrame, top_k: int = 6) -> List[dict]:
        sv = explainer.shap_values(x_row)[0]
        contribs = sorted(
            ({"feature": f, "shap": float(v)} for f, v in zip(feature_names, sv)),
            key=lambda d: abs(d["shap"]), reverse=True,
        )
        return contribs[:top_k]

    return explain


# ----------------------------- main -----------------------------

def main() -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    log("Loading data + models...")
    df = load_dataset(DATA_PATH)
    x, y, feature_names, cat_cols = prepare_features(df)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_SEED,
    )
    y_train_log = np.log1p(y_train)

    # Rebuild the make_model target-encoding feature exactly as training did:
    # from the proper-train slice (same split seed), applied to train and test.
    from train_price_model import (
        CALIBRATION_FRACTION, ENCODING_FEATURE,
        apply_make_model_encoding, compute_make_model_encoding,
    )
    x_ptrain, _, y_ptrain, _ = train_test_split(
        x_train, y_train, test_size=CALIBRATION_FRACTION, random_state=RANDOM_SEED,
    )
    mm_enc, mm_prior = compute_make_model_encoding(x_ptrain, np.log1p(y_ptrain))
    x_train = apply_make_model_encoding(x_train, mm_enc, mm_prior)
    x_test = apply_make_model_encoding(x_test, mm_enc, mm_prior)
    feature_names = feature_names + [ENCODING_FEATURE]
    actual = y_test.to_numpy()
    # Keep the original processed columns aligned to the test rows for segmenting.
    test_df = df.loc[x_test.index]

    models = load_models()

    # ---- point + quantile predictions (sorted to prevent crossing) ----
    # Use the same Mondrian (price-segment-specific) conformal adjustment served by
    # predict_listing.py. Older artifacts without it fall back to one global delta.
    pred_log_q10 = models["q10"].predict(x_test)
    pred_log_q50 = models["q50"].predict(x_test)
    pred_log_q90 = models["q90"].predict(x_test)
    mondrian = conformal_mondrian.load(ARTIFACT_DIR)
    if mondrian is not None:
        lo_delta, hi_delta = conformal_mondrian.deltas_for(mondrian, pred_log_q50)
    else:
        global_delta = conformal.load_delta(ARTIFACT_DIR)
        lo_delta = hi_delta = global_delta

    pred_pt = predict_dollars(models["mean"], x_test)
    band_lo_log, median_log, band_hi_log = conformal.calibrated_band_logs(
        pred_log_q10, pred_log_q50, pred_log_q90, lo_delta, hi_delta,
    )
    q = {
        "q10": np.expm1(band_lo_log),
        "q50": np.expm1(median_log),
        "q90": np.expm1(band_hi_log),
    }
    raw_q = {
        "q10": np.expm1(pred_log_q10),
        "q50": np.expm1(pred_log_q50),
        "q90": np.expm1(pred_log_q90),
    }

    # ---- baselines ----
    log("\n=== Baselines vs model (fixed evaluation set, $ space) ===")
    results = {
        "model_point": metric_block(actual, pred_pt),
        "baseline_global_median": metric_block(actual, baseline_global_median(y_train, len(actual))),
        "baseline_group_median": metric_block(actual, baseline_group_median(x_train, x_test, y_train)),
        "baseline_linear_depreciation": metric_block(actual, baseline_linear_depreciation(x_train, y_train_log, x_test)),
    }
    for name, m in results.items():
        log(f"  {name:32s} MAE=${m['mae']:>8,.0f}  R2={m['r2']:.3f}  MAPE={m['mape_percent']:.1f}%")

    # ---- segment analysis ----
    log("\n=== Segment error analysis ===")
    seg = segment_errors(test_df, actual, pred_pt, q["q10"], q["q90"])
    seg.to_csv(EVAL_DIR / "segment_errors.csv", index=False)
    log(seg.to_string(index=False))

    # ---- calibration ----
    log("\n=== Raw quantile diagnostics + adjusted interval coverage ===")
    calib = quantile_calibration(actual, raw_q)
    interval_cov = float(np.mean((actual >= q["q10"]) & (actual <= q["q90"])))
    calib.to_csv(EVAL_DIR / "calibration.csv", index=False)
    plot_calibration(calib, EVAL_DIR / "calibration_curve.png")
    log(calib.to_string(index=False))
    log(f"  [q10,q90] interval coverage: {interval_cov*100:.1f}% (target ~80%)")

    # ---- SHAP ----
    log("\n=== SHAP global importance ===")
    fi = shap_global(models, x_test, feature_names)
    fi.to_csv(EVAL_DIR / "shap_global.csv", index=False)
    log(fi.head(10).to_string(index=False))

    explain = make_listing_explainer(models, feature_names, cat_cols)
    example_contribs = explain(x_test.iloc[[0]])
    log("\nExample per-listing explanation (row 0, log-price contributions):")
    for c in example_contribs:
        log(f"    {c['feature']:20s} {c['shap']:+.3f}")

    # ---- write a human-readable report ----
    write_report(results, seg, calib, interval_cov, fi)
    log(f"\nSaved evaluation outputs to {EVAL_DIR}/ and a report at {EVAL_DIR/'evaluation_report.md'}")


def write_report(results, seg, calib, interval_cov, fi) -> None:
    fmt = lambda v: f"${v:,.0f}"
    m = results["model_point"]
    gm = results["baseline_global_median"]
    grp = results["baseline_group_median"]
    lin = results["baseline_linear_depreciation"]

    lines = [
        "# Model Evaluation Report",
        "",
        "_Market-relative vehicle pricing. Target = asking price from listings (not realized",
        "sale price); all verdicts are relative to comparable listings on the market._",
        "",
        "## 1. Model vs baselines (fixed evaluation split)",
        "",
        "| Method | MAE | R² | MAPE |",
        "|---|---|---|---|",
        f"| **LightGBM (point)** | {fmt(m['mae'])} | {m['r2']:.3f} | {m['mape_percent']:.1f}% |",
        f"| Global-median baseline | {fmt(gm['mae'])} | {gm['r2']:.3f} | {gm['mape_percent']:.1f}% |",
        f"| Same make/model median | {fmt(grp['mae'])} | {grp['r2']:.3f} | {grp['mape_percent']:.1f}% |",
        f"| Linear depreciation (age+miles+luxury) | {fmt(lin['mae'])} | {lin['r2']:.3f} | {lin['mape_percent']:.1f}% |",
        "",
        "The relevant question is not only whether R² is high, but whether the model beats",
        "a simple comparable-vehicle lookup: the median price of the same make/model.",
        "",
        "## 2. Where the model fails (segment analysis)",
        "",
        "See `segment_errors.csv`. Error is not uniform: it concentrates in high-price /",
        "luxury / older segments, where comparable listings are sparse.",
        "",
        "```",
        seg.to_string(index=False),
        "```",
        "",
        "## 3. Is the confidence band honest? (calibration)",
        "",
        f"[q10, q90] interval coverage on the evaluation split: **{interval_cov*100:.1f}%** (target ~80%).",
        "Raw quantile diagnostics (`calibration_curve.png`):",
        "",
        "```",
        calib.to_string(index=False),
        "```",
        "",
        "## 4. What drives price (SHAP)",
        "",
        "Global importances (`shap_global.png`, `shap_global.csv`). Top features:",
        "",
        "```",
        fi.head(10).to_string(index=False),
        "```",
        "",
        "The reusable `make_listing_explainer` function returns signed contributions",
        "for one listing in log-price space.",
        "",
        "## Known limitations",
        "",
        "- **Asking, not sale price.** We never observe what a car actually sold for.",
        "- **Data scale + coverage.** The six-figure/luxury segment is thinly represented,",
        "  and this is where error concentrates; collecting more high-price and rare-model",
        "  listings is the top driver of future accuracy.",
        "- **Temporal snapshot.** Prices drift; the model reflects one collection window.",
        "- **Dataset-wide preprocessing.** Price-based outlier rules and rare-category",
        "  grouping based on full-sample counts were applied before the 80/20 split.",
        "- **Repeated evaluation use.** The fixed split was reviewed across versions 10-13",
        "  and influenced feature selection during development. Its rows were not used to",
        "  fit or calibrate version 13, but this is not a one-time independent final test.",
    ]
    (EVAL_DIR / "evaluation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
