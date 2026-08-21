"""
Robustness & generalization study.

Two questions a skeptic asks about the headline single-split numbers:

  1. Is the accuracy an artifact of one lucky test split? -> Repeated K-fold CV
     shows how out-of-fold MAE varies across several train-test splits rather
     than relying only on the seed-42 split.

  2. Does the model actually generalize, or does it lean on having seen each
     make_model before? -> make_model-GROUPED K-fold holds every listing of a
     model line out together, so the test folds contain only *unseen* model lines
     and the make_model target encoding must fall back to its prior. The gap
     between random-fold and grouped-fold MAE is the price of an unseen model line.

Leakage control: the make_model target encoding is re-fit inside each fold's
training rows only, never on the full data, so no fold's test rows inform their own
encoding. Models use the production monotone constraints.

Standalone: python3 -m experiments.run_robustness   (from src/)
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, RepeatedKFold

import paths
import train_price_model as tpm
from experiments.models import LGBM_PARAMS
from experiments.stats import mape as mape_fn

OUT_DIR = Path(paths.ARTIFACT_DIR) / "paper"

N_SPLITS = 5
N_REPEATS = 3
SEED = 42


def _load_preencoding():
    """x (with make_model, no encoding), y, and the feature order used at fit time."""
    df = tpm.load_dataset(tpm.DATA_PATH)
    x, y, feats, cats = tpm.prepare_features(df)
    feature_order = feats + [tpm.ENCODING_FEATURE]
    return x.reset_index(drop=True), y.reset_index(drop=True), cats, feature_order


def _fit_fold(x_tr, y_tr, x_te, cats, feature_order):
    """Refit encoding on THIS fold's train rows, fit the monotone point model, predict."""
    enc, prior = tpm.compute_make_model_encoding(x_tr, np.log1p(y_tr))
    x_tr_e = tpm.apply_make_model_encoding(x_tr, enc, prior)[feature_order]
    x_te_e = tpm.apply_make_model_encoding(x_te, enc, prior)[feature_order]
    params = dict(LGBM_PARAMS)
    params["monotone_constraints"] = tpm.monotone_constraints_for(feature_order)
    m = tpm.make_lgbm("regression", params=params)
    m.fit(x_tr_e, np.log1p(y_tr), categorical_feature=cats if cats else "auto")
    return np.expm1(m.predict(x_te_e))


def _cv_maes(x, y, cats, feature_order, splitter, groups=None):
    maes = []
    for tr, te in splitter.split(x, y, groups):
        pred = _fit_fold(x.iloc[tr], y.iloc[tr], x.iloc[te], cats, feature_order)
        maes.append(mean_absolute_error(y.iloc[te].to_numpy(), pred))
    return np.array(maes)


def run() -> dict:
    x, y, cats, feature_order = _load_preencoding()

    # 1) Repeated K-fold -> distribution of MAE across several splits.
    rkf = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    rand_maes = _cv_maes(x, y, cats, feature_order, rkf)

    # 2) make_model-grouped K-fold -> every test fold is unseen model lines.
    groups = x["make_model"].astype(str).to_numpy()
    gkf = GroupKFold(n_splits=N_SPLITS)
    group_maes = _cv_maes(x, y, cats, feature_order, gkf, groups=groups)

    n_models = int(len(np.unique(groups)))
    results = {
        "n_rows": int(len(x)),
        "n_make_models": n_models,
        "repeated_kfold": {
            "n_splits": N_SPLITS, "n_repeats": N_REPEATS,
            "mae_mean": float(rand_maes.mean()),
            "mae_ci_lo": float(np.percentile(rand_maes, 2.5)),
            "mae_ci_hi": float(np.percentile(rand_maes, 97.5)),
            "mae_std": float(rand_maes.std()),
        },
        "grouped_kfold": {
            "n_splits": N_SPLITS,
            "mae_mean": float(group_maes.mean()),
            "mae_std": float(group_maes.std()),
        },
    }
    results["unseen_model_penalty"] = (
        results["grouped_kfold"]["mae_mean"] - results["repeated_kfold"]["mae_mean"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "robustness.json").write_text(json.dumps(results, indent=2))
    _print(results)
    return results


def _print(r: dict) -> None:
    rk, gk = r["repeated_kfold"], r["grouped_kfold"]
    print(f"\n=== Robustness (n={r['n_rows']:,}, {r['n_make_models']} make_models) ===")
    print(f"Repeated {rk['n_splits']}-fold x{rk['n_repeats']}  MAE ${rk['mae_mean']:,.0f} "
          f"[{rk['mae_ci_lo']:,.0f}, {rk['mae_ci_hi']:,.0f}] (empirical 2.5th-97.5th percentile range across folds)")
    print(f"make_model-grouped {gk['n_splits']}-fold (unseen lines)  MAE "
          f"${gk['mae_mean']:,.0f} (±{gk['mae_std']:,.0f})")
    print(f"unseen-model penalty: +${r['unseen_model_penalty']:,.0f} MAE")


if __name__ == "__main__":
    run()
