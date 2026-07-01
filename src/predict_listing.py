"""
Predict a fair price band for one car.

Run after train_price_model.py has produced model_artifacts/.

Two ways to use it:

1. From the command line, edit the EXAMPLE_LISTING below and run:
       python3 predict_listing.py

2. From other code:
       from predict_listing import evaluate
       result = evaluate({...})

This is the seed for the "is this listing over/underpriced?" feature.

Notes on correctness:
- The model learns from *asking* prices on listings, not actual sale prices. So a
  verdict of "deal / fair / overpriced" means priced low / in-line / high *relative to
  comparable listings on the market* — not relative to a car's true resale value.
- The three quantile models (q10/q50/q90) are trained independently and can cross for
  some inputs. We sort them per-prediction so the band is always q10 <= q50 <= q90.
- If a listing's make_model or trim was never seen in training, the prediction is far
  less reliable; we surface that as a warning rather than returning a confident number.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import paths


ARTIFACT_DIR = paths.ARTIFACT_DIR
MODEL_FILES = {"q10": "lgbm_q10.pkl", "q50": "lgbm_q50.pkl", "q90": "lgbm_q90.pkl", "mean": "lgbm_mean.pkl"}

# Categorical features for which an unseen value meaningfully degrades the estimate.
# (make_model and trim carry most of the signal; an unseen region/color matters less.)
HIGH_IMPACT_CATEGORICALS = ("make_model", "trim")

# Module-level cache so the 4 pickles + metadata load once, not on every call.
_CACHE: Optional[dict] = None


def _load() -> dict:
    """Load models + metadata once and cache them for the process lifetime."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    with open(ARTIFACT_DIR / "feature_names.json") as f:
        feature_names = json.load(f)
    with open(ARTIFACT_DIR / "cat_features.json") as f:
        cat_features = json.load(f)

    # category_levels.json is optional (older artifacts may predate it).
    category_levels: Dict[str, List] = {}
    levels_path = ARTIFACT_DIR / "category_levels.json"
    if levels_path.exists():
        with open(levels_path) as f:
            category_levels = json.load(f)

    models = {}
    for name, fname in MODEL_FILES.items():
        with open(ARTIFACT_DIR / fname, "rb") as f:
            models[name] = pickle.load(f)

    _CACHE = {
        "models": models,
        "feature_names": feature_names,
        "cat_features": cat_features,
        "category_levels": category_levels,
    }
    return _CACHE


def _row_from_listing(listing: Dict, feature_names, cat_features) -> pd.DataFrame:
    """Build a 1-row DataFrame in the exact column order the model expects."""
    data = {col: [listing.get(col, np.nan)] for col in feature_names}
    df = pd.DataFrame(data)
    for c in cat_features:
        if c in df.columns:
            df[c] = df[c].astype("category")
    for c in df.columns:
        if c not in cat_features:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _unseen_category_warnings(listing: Dict, category_levels: Dict[str, List]) -> List[str]:
    """Flag high-impact categorical values the model never saw in training."""
    warnings: List[str] = []
    for col in HIGH_IMPACT_CATEGORICALS:
        known = category_levels.get(col)
        if not known:
            continue
        value = listing.get(col)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        if value not in known:
            warnings.append(
                f"'{value}' is an unseen {col}; the model has no comparable listings "
                f"for it, so this estimate is low-confidence."
            )
    return warnings


def evaluate(listing: Dict, asking_price: Optional[float] = None) -> Dict:
    """
    listing: dict of car features. Use the same column names as the processed CSV.
             Missing fields are fine — the model handles NaN.
    asking_price: if given, returns a verdict (deal / fair / overpriced).
    """
    cache = _load()
    models = cache["models"]
    x = _row_from_listing(listing, cache["feature_names"], cache["cat_features"])

    pred = {name: float(np.expm1(m.predict(x)[0])) for name, m in models.items()}

    # Enforce a non-crossing band: q10 <= q50 <= q90 even if the independently
    # trained quantile models disagree for this particular input.
    band_lo, median, band_hi = sorted([pred["q10"], pred["q50"], pred["q90"]])
    point = pred["mean"]

    warnings = _unseen_category_warnings(listing, cache["category_levels"])

    out = {
        "predicted_price": round(point),
        "median_estimate": round(median),
        "fair_range_low": round(band_lo),
        "fair_range_high": round(band_hi),
        "low_confidence": bool(warnings),
        "warnings": warnings,
    }

    if asking_price is not None:
        ap = float(asking_price)
        if ap > band_hi:
            verdict = "overpriced"
        elif ap < band_lo:
            verdict = "deal"
        else:
            verdict = "fair"
        delta = ap - point
        pct = 100 * delta / point if point else 0
        out.update({
            "asking_price": ap,
            "delta_vs_predicted": round(delta),
            "delta_vs_predicted_percent": round(pct, 1),
            "verdict": verdict,
        })

    return out


def pretty_print(result: Dict) -> None:
    print("\n--- Price Estimate ---")
    print(f"Predicted price:   ${result['predicted_price']:>9,}")
    print(f"Median (q50):      ${result['median_estimate']:>9,}")
    print(f"Fair range:        ${result['fair_range_low']:>9,} – ${result['fair_range_high']:,}")
    if "verdict" in result:
        print(f"Asking price:      ${int(result['asking_price']):>9,}")
        sign = "+" if result["delta_vs_predicted"] >= 0 else ""
        print(f"Delta vs predict:  {sign}${result['delta_vs_predicted']:,} ({sign}{result['delta_vs_predicted_percent']}%)")
        print(f"VERDICT:           {result['verdict'].upper()}")
    if result.get("warnings"):
        print("\n  ⚠ low-confidence:")
        for w in result["warnings"]:
            print(f"    - {w}")
    print()


# ----------------------------------------------------------------
# Edit this and run `python3 predict_listing.py` to try it out.
# Only the fields you have are needed; the rest can be omitted.
# ----------------------------------------------------------------
EXAMPLE_LISTING = {
    "make_model": "honda__civic",
    "year": 2020,
    "mileage": 42000,
    "car_age": 6,
    "miles_per_year": 7000,
    "trim": "sport",
    "body_style": "sedan",
    "mileage_bucket": "low",
    "is_luxury_make": 0,
    "is_new": 0,
    "drivetrain": "fwd",
    "fuel_type_group": "gas",
    "transmission_type": "cvt",
    "car_color": "black",
    "title_status": "clean",
    "accident_history": "no_accident",
    "owner_count": 1,
    "cpo": 0,
    "engine_size_l": 2.0,
    "mpg": 35,
    "region": "Issaquah, WA",
}
EXAMPLE_ASKING_PRICE = 19500


if __name__ == "__main__":
    result = evaluate(EXAMPLE_LISTING, asking_price=EXAMPLE_ASKING_PRICE)
    pretty_print(result)
