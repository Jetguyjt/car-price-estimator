"""
Ablation study: the marginal contribution of each modeling component to point
accuracy, all on the identical held-out test set with bootstrap 95% CIs.

Two human baselines anchor the bottom (what you'd do without a model), then
components are added one at a time:

    global-median baseline
    same make/model median baseline
    LightGBM, raw-price target, base features
      + log-price target
      + trim_tier feature
      + make_model target encoding      (= the production point model)

Each row reports MAE / RMSE / R2 / MAPE with a 95% CI, and (for the model rows)
luxury-segment MAE, so the reader can see where each component pays off.

Note on scope: trim_tier and make_model encoding are toggled here on a fixed
dataset. The data-pipeline fixes (price-cap, $/mile scoping, split-model repair)
change which rows exist, so they are reported separately as a data-quality case
study, not as same-split ablation rows.

Standalone: python3 -m experiments.run_ablation   (from src/)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import paths
import train_price_model as tpm
from experiments import data as D
from experiments.models import fit_point
from experiments.stats import METRICS, bootstrap_ci

OUT_DIR = Path(paths.ARTIFACT_DIR) / "paper"

BASE_EXCLUDE = ("trim_tier", tpm.ENCODING_FEATURE)


def _metrics_with_ci(actual, pred, luxury_mask=None) -> dict:
    out = {}
    for m in ("MAE", "RMSE", "R2", "MAPE"):
        point, lo, hi = bootstrap_ci(actual, pred, m)
        out[m] = {"point": point, "lo": lo, "hi": hi}
    if luxury_mask is not None and luxury_mask.sum() > 0:
        out["MAE_luxury"] = {"point": float(METRICS["MAE"](
            actual[luxury_mask], pred[luxury_mask])), "n": int(luxury_mask.sum())}
    return out


def _baseline_global_median(d) -> np.ndarray:
    return np.full(len(d.y_test), float(np.median(d.y_train)))


def _baseline_group_median(d) -> np.ndarray:
    gm = float(np.median(d.y_train))
    grp = d.y_train.groupby(d.x_train["make_model"].astype(str), observed=True).median()
    keys = d.x_test["make_model"].astype(str)
    return keys.map(grp).fillna(gm).to_numpy()


def run() -> dict:
    d = D.load()
    luxury = d.test_meta["is_luxury"].astype(int).to_numpy().astype(bool)
    actual = d.y_test.to_numpy()

    base_feats = [f for f in d.feature_names if f not in BASE_EXCLUDE]
    plus_tier = base_feats + ["trim_tier"]
    full = plus_tier + [tpm.ENCODING_FEATURE]

    rows = []

    def add(name, pred, is_model=True):
        rows.append({"name": name,
                     "metrics": _metrics_with_ci(actual, pred,
                                                 luxury if is_model else None)})

    add("baseline: global median", _baseline_global_median(d), is_model=False)
    add("baseline: make/model median", _baseline_group_median(d), is_model=False)
    add("LightGBM (raw target, base feats)",
        fit_point(d, base_feats, log_target=False))
    add("+ log-price target", fit_point(d, base_feats, log_target=True))
    add("+ trim_tier", fit_point(d, plus_tier, log_target=True))
    add("+ make_model encoding (full)", fit_point(d, full, log_target=True))

    results = {"n_test": int(len(actual)), "n_test_luxury": int(luxury.sum()),
               "rows": rows}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "ablation.json").write_text(json.dumps(results, indent=2))
    _print(results)
    return results


def _print(r: dict) -> None:
    print(f"\n=== Ablation (n_test={r['n_test']}, luxury={r['n_test_luxury']}) ===")
    print(f"{'component':<38}{'MAE':>18}{'R2':>16}{'MAPE':>16}{'lux MAE':>10}")
    for row in r["rows"]:
        m = row["metrics"]
        mae = f"${m['MAE']['point']:,.0f}"
        r2 = f"{m['R2']['point']:.3f}"
        mape = f"{m['MAPE']['point']:.1f}%"
        lux = f"${m['MAE_luxury']['point']:,.0f}" if "MAE_luxury" in m else "-"
        print(f"{row['name']:<38}{mae:>18}{r2:>16}{mape:>16}{lux:>10}")


if __name__ == "__main__":
    run()
