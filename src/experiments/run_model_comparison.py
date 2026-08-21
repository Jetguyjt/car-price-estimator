"""
Model comparison: LightGBM vs standard regressors on identical features, split,
and target (log-price), with bootstrap 95% CIs and a paired-bootstrap
significance test of every comparator against LightGBM.

This answers the reviewer's "why this model?" - not by assertion but by a
controlled comparison where only the learner changes. XGBoost is included when
installed; it and LightGBM are the two gradient-boosted baselines, with a linear
model and a random forest bracketing the range.

Standalone: python3 -m experiments.run_model_comparison   (from src/)
"""

import json
from pathlib import Path

import numpy as np

import paths
import train_price_model as tpm
from experiments import data as D
from experiments.models import fit_point
from experiments.stats import bootstrap_ci, paired_bootstrap_test

OUT_DIR = Path(paths.ARTIFACT_DIR) / "paper"

CANDIDATES = [
    ("Ridge (linear)", "ridge"),
    ("Random forest", "random_forest"),
    ("HistGradientBoosting", "hist_gbm"),
    ("XGBoost", "xgboost"),
    ("LightGBM (ours)", "lgbm"),
]


def _available(model: str) -> bool:
    if model == "xgboost":
        try:
            import xgboost  # noqa: F401
            return True
        except ImportError:
            return False
    return True


def run() -> dict:
    d = D.load()
    actual = d.y_test.to_numpy()
    feats = d.feature_names

    preds, rows = {}, []
    for label, key in CANDIDATES:
        if not _available(key):
            continue
        pred = fit_point(d, feats, log_target=True, model=key)
        preds[label] = pred
        entry = {"model": label, "key": key, "metrics": {}}
        for m in ("MAE", "RMSE", "R2", "MAPE"):
            point, lo, hi = bootstrap_ci(actual, pred, m)
            entry["metrics"][m] = {"point": point, "lo": lo, "hi": hi}
        rows.append(entry)

    # Paired tests: each comparator vs LightGBM on MAE (shared resamples).
    ours = preds["LightGBM (ours)"]
    for entry in rows:
        if entry["model"] == "LightGBM (ours)":
            entry["vs_lgbm"] = None
            continue
        entry["vs_lgbm"] = paired_bootstrap_test(actual, preds[entry["model"]], ours, "MAE")

    results = {"n_test": int(len(actual)), "features": feats, "rows": rows}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "model_comparison.json").write_text(json.dumps(results, indent=2))
    _print(results)
    return results


def _print(r: dict) -> None:
    print(f"\n=== Model comparison (n_test={r['n_test']}) ===")
    print(f"{'model':<24}{'MAE (95% CI)':>26}{'R2':>10}{'MAPE':>9}{'vs LGBM ΔMAE (p)':>22}")
    for e in sorted(r["rows"], key=lambda x: x["metrics"]["MAE"]["point"], reverse=True):
        m = e["metrics"]
        mae = f"${m['MAE']['point']:,.0f} [{m['MAE']['lo']:,.0f},{m['MAE']['hi']:,.0f}]"
        r2 = f"{m['R2']['point']:.3f}"
        mape = f"{m['MAPE']['point']:.1f}%"
        if e["vs_lgbm"]:
            v = e["vs_lgbm"]
            cmp = f"+${v['diff']:,.0f} (p={v['p_value']:.3f})"
        else:
            cmp = "-"
        print(f"{e['model']:<24}{mae:>26}{r2:>10}{mape:>9}{cmp:>22}")


if __name__ == "__main__":
    run()
