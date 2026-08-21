"""
Monotonicity study: does the model obey basic economics, and what does enforcing it cost?

An unconstrained gradient-boosted price model has no reason to be monotone in mileage
or age - and it isn't: on a ceteris-paribus counterfactual sweep it raises the predicted
price of a large fraction of cars when only their mileage increases, and lowers it when
only their model year increases. Both are nonsensical to any buyer.

This study:
  1. Audits violations by counterfactual sweep - hold every other feature fixed, move one
     feature, count cars whose prediction moves the wrong way - for unconstrained vs
     monotone-constrained LightGBM.
  2. Reports the point-accuracy cost of the constraints (MAE/R²/MAPE with bootstrap CIs),
     showing the sanity guarantee is ~free.

Standalone: python3 -m experiments.run_monotonicity   (from src/)
"""

import json
from pathlib import Path

import numpy as np

import paths
import train_price_model as tpm
from experiments import data as D
from experiments.models import LGBM_PARAMS, _prep, fit_point
from experiments.stats import METRICS, bootstrap_ci

OUT_DIR = Path(paths.ARTIFACT_DIR) / "paper"

# Counterfactual sweeps: (feature, deltas, wrong-direction sign). sign=+1 means a
# violation is the prediction going UP as the feature increases (mileage/age/owners);
# sign=-1 means a violation is the prediction going DOWN (year).
# Only clean, realistic sweeps: mileage moves with miles_per_year kept consistent;
# year moves with car_age kept consistent. (Sweeping car_age while holding year fixed
# is physically impossible - car_age = current_year - year - so it isn't audited.)
SWEEPS = {
    "mileage": {"deltas": [0, 20_000, 40_000, 60_000], "violation_when": "up"},
    "year": {"deltas": [0, 1, 2, 3], "violation_when": "down"},
}
VIOLATION_TOL = 1.0    # dollars; ignore sub-$1 numerical noise


def _fit_mean(data, monotone: bool):
    x_tr, cats = _prep(data.x_train, data.feature_names, data.cat_cols)
    params = dict(LGBM_PARAMS)
    if monotone:
        params["monotone_constraints"] = tpm.monotone_constraints_for(x_tr.columns.tolist())
    m = tpm.make_lgbm("regression", params=params)
    m.fit(x_tr, data.y_train_log, categorical_feature=cats if cats else "auto")
    return m, cats


def _sweep_violation_rate(model, base_x, feature, deltas, violation_when) -> dict:
    """Fraction of rows whose prediction moves the wrong way as `feature` increases,
    and the mean worst wrong-way move in dollars."""
    # Ceteris paribus: move only `feature`, hold everything else (including the
    # collinear derived columns) fixed - this is exactly what a monotone constraint
    # guarantees, so a correctly-constrained model shows zero violations here.
    preds = []
    for dv in deltas:
        x = base_x.copy()
        x[feature] = x[feature].astype(float) + dv
        preds.append(np.expm1(model.predict(x)))
    preds = np.array(preds)                       # [len(deltas), n]
    steps = np.diff(preds, axis=0)                # consecutive changes
    wrong = steps if violation_when == "up" else -steps
    any_violation = np.any(wrong > VIOLATION_TOL, axis=0)
    worst = np.maximum(np.max(wrong, axis=0), 0)
    return {"violation_rate": float(np.mean(any_violation)),
            "mean_worst_violation_dollars": float(np.mean(worst)),
            "n": int(preds.shape[1])}


def run() -> dict:
    d = D.load()
    actual = d.y_test.to_numpy()
    base_x, _ = _prep(d.x_test, d.feature_names, d.cat_cols)
    base_x = base_x.reset_index(drop=True).copy()

    audits = {}
    for monotone in (False, True):
        model, _ = _fit_mean(d, monotone)
        key = "constrained" if monotone else "unconstrained"
        audits[key] = {f: _sweep_violation_rate(model, base_x, f, s["deltas"],
                                                 s["violation_when"])
                       for f, s in SWEEPS.items()}

    # Accuracy cost of the constraints, with bootstrap CIs.
    accuracy = {}
    for monotone in (False, True):
        pred = fit_point(d, d.feature_names, log_target=True, monotone=monotone)
        key = "constrained" if monotone else "unconstrained"
        accuracy[key] = {}
        for m in ("MAE", "R2", "MAPE"):
            point, lo, hi = bootstrap_ci(actual, pred, m)
            accuracy[key][m] = {"point": point, "lo": lo, "hi": hi}

    results = {"n_test": int(len(actual)), "audits": audits, "accuracy": accuracy,
               "constraints": {c: d_ for c, d_ in tpm.MONOTONE_DIRECTIONS.items()}}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "monotonicity.json").write_text(json.dumps(results, indent=2))
    _print(results)
    return results


def _print(r: dict) -> None:
    print(f"\n=== Monotonicity audit (n_test={r['n_test']}) ===")
    print(f"{'sweep':<12}{'unconstrained viol.':>22}{'constrained viol.':>22}")
    for f in SWEEPS:
        u = r["audits"]["unconstrained"][f]["violation_rate"]
        c = r["audits"]["constrained"][f]["violation_rate"]
        print(f"{f:<12}{u:>21.1%}{c:>22.1%}")
    print("\naccuracy cost:")
    for key in ("unconstrained", "constrained"):
        a = r["accuracy"][key]
        print(f"  {key:<14} MAE ${a['MAE']['point']:,.0f} "
              f"[{a['MAE']['lo']:,.0f},{a['MAE']['hi']:,.0f}]  "
              f"R2 {a['R2']['point']:.3f}  MAPE {a['MAPE']['point']:.1f}%")


if __name__ == "__main__":
    run()
