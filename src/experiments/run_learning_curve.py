"""
Learning-curve / data-scaling study - quantifies what more data would buy.

Trains on increasing subsamples of the pool and measures test MAE (overall and
luxury), then fits a power law  MAE(n) = a * n^(-b) + c  and extrapolates to
50k / 100k / 175k rows. This turns "we should scrape more" from an assertion into
a number: the projected MAE at the planned collection scale, and the irreducible
error floor c that more data cannot beat.

Leakage control: the make_model target encoding is re-fit *inside* each subsample
(never on the full pool), and models use the production monotone constraints, so
each point is a clean "trained on n rows" estimate.

Standalone: python3 -m experiments.run_learning_curve   (from src/)
"""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error

import paths
import train_price_model as tpm
from experiments import data as D
from experiments.models import LGBM_PARAMS
from experiments.stats import mape as mape_fn

OUT_DIR = Path(paths.ARTIFACT_DIR) / "paper"

SIZES = [1000, 2000, 4000, 6000, 8000]
SEEDS = [0, 1, 2, 3, 4]
EXTRAPOLATE_TO = [50_000, 100_000, 175_000]


def _fit_and_score(x_pool, y_pool, cats, feature_order, x_test, y_test, lux_mask, n, seed):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x_pool), size=n, replace=False)
    x_sub = x_pool.iloc[idx].copy()
    y_sub = y_pool.iloc[idx].copy()

    # Re-fit the make_model encoding on THIS subsample only (leakage-safe).
    enc, prior = tpm.compute_make_model_encoding(x_sub, np.log1p(y_sub))
    x_sub_e = tpm.apply_make_model_encoding(x_sub, enc, prior)[feature_order]
    x_test_e = tpm.apply_make_model_encoding(x_test, enc, prior)[feature_order]

    params = dict(LGBM_PARAMS)
    params["monotone_constraints"] = tpm.monotone_constraints_for(feature_order)
    m = tpm.make_lgbm("regression", params=params)
    m.fit(x_sub_e, np.log1p(y_sub), categorical_feature=cats if cats else "auto")
    pred = np.expm1(m.predict(x_test_e))
    y_test_arr = y_test.to_numpy()
    return {
        "mae": float(mean_absolute_error(y_test_arr, pred)),
        "mape": float(mape_fn(y_test_arr, pred)),
        "mae_luxury": float(mean_absolute_error(y_test_arr[lux_mask], pred[lux_mask]))
        if lux_mask.sum() else float("nan"),
    }


def _power_law(n, a, b, c):
    return a * np.power(n, -b) + c


def run() -> dict:
    d = D.load(with_encoding=False)   # pool without encoding; we re-fit per subsample
    feature_order = d.feature_names + [tpm.ENCODING_FEATURE]
    lux_mask = d.test_meta["is_luxury"].astype(int).to_numpy().astype(bool)

    sizes = [s for s in SIZES if s < len(d.x_train)] + [len(d.x_train)]
    curve = []
    for n in sizes:
        runs = [_fit_and_score(d.x_train, d.y_train, d.cat_cols, feature_order,
                               d.x_test, d.y_test, lux_mask, n, s) for s in SEEDS]
        maes = np.array([r["mae"] for r in runs])
        lux = np.array([r["mae_luxury"] for r in runs])
        curve.append({"n": n, "mae_mean": float(maes.mean()), "mae_std": float(maes.std()),
                      "mae_luxury_mean": float(np.nanmean(lux))})

    ns = np.array([c["n"] for c in curve], float)
    maes = np.array([c["mae_mean"] for c in curve], float)
    extrap = {}
    fit_params = None
    try:
        popt, _ = curve_fit(_power_law, ns, maes,
                            p0=[maes[0] * ns[0] ** 0.3, 0.3, maes[-1] * 0.8],
                            maxfev=20000, bounds=([0, 0, 0], [np.inf, 2, maes.min()]))
        fit_params = {"a": float(popt[0]), "b": float(popt[1]), "c_floor": float(popt[2])}
        extrap = {str(N): float(_power_law(N, *popt)) for N in EXTRAPOLATE_TO}
    except Exception as e:  # pragma: no cover - fit robustness
        fit_params = {"error": str(e)}

    results = {"sizes": sizes, "curve": curve, "power_law": fit_params,
               "extrapolation": extrap, "current_full_n": int(len(d.x_train)),
               "n_test": int(len(d.x_test))}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "learning_curve.json").write_text(json.dumps(results, indent=2))
    _print(results)
    return results


def _print(r: dict) -> None:
    print(f"\n=== Learning curve (test n={r['n_test']}) ===")
    print(f"{'train n':>9}{'MAE':>10}{'±std':>8}{'luxury MAE':>13}")
    for c in r["curve"]:
        print(f"{c['n']:>9,}{c['mae_mean']:>10,.0f}{c['mae_std']:>8,.0f}"
              f"{c['mae_luxury_mean']:>13,.0f}")
    if "a" in r["power_law"]:
        p = r["power_law"]
        print(f"\npower law  MAE(n) = {p['a']:,.0f}*n^-{p['b']:.3f} + {p['c_floor']:,.0f} "
              f"(irreducible floor ~${p['c_floor']:,.0f})")
        print("extrapolated MAE:")
        for N, v in r["extrapolation"].items():
            print(f"  {int(N):>7,} rows -> ${v:,.0f}")


if __name__ == "__main__":
    run()
