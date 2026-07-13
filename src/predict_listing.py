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
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import paths
import conformal


ARTIFACT_DIR = paths.ARTIFACT_DIR
MODEL_FILES = {"q10": "lgbm_q10.pkl", "q50": "lgbm_q50.pkl", "q90": "lgbm_q90.pkl", "mean": "lgbm_mean.pkl"}

# Categorical features for which an unseen value meaningfully degrades the estimate.
# (make_model and trim carry most of the signal; an unseen region/color matters less.)
HIGH_IMPACT_CATEGORICALS = ("make_model", "trim")

# Module-level cache so the 4 pickles + metadata load once, not on every call.
_CACHE: Optional[dict] = None


def assemble_listing(
    make: str,
    model: str,
    year,
    mileage,
    *,
    trim=None,
    body_style=None,
    drivetrain=None,
    transmission_type=None,
    fuel_type_group=None,
    car_color=None,
    title_status=None,
    accident_history=None,
    engine_size_l=None,
    mpg=None,
    owner_count=None,
    cpo=None,
    region="unknown",
    is_new=False,
) -> Dict:
    """Turn user-entered primary attributes into the full feature dict the model
    expects, deriving the engineered columns (make_model, car_age, miles_per_year,
    mileage_bucket, is_luxury_make) exactly as the training pipeline does.

    Kept here (not in the app) so the derivation is reusable and unit-testable, and
    so it shares the luxury/mileage-bucket definitions with preprocess_cars_data.
    """
    import preprocess_cars_data as pp

    make_l = (make or "").strip().lower()
    model_l = (model or "").strip().lower()

    year_val = pd.to_numeric(year, errors="coerce")
    mileage_val = pd.to_numeric(mileage, errors="coerce")

    current_year = datetime.now().year
    car_age = None
    if pd.notna(year_val):
        car_age = max(current_year - int(year_val), 0)

    miles_per_year = None
    if pd.notna(mileage_val) and car_age is not None:
        miles_per_year = float(mileage_val) / max(car_age, 1)

    def _norm(v):
        # Blank strings become None so unseen-category logic and the model see NaN.
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None

    return {
        "make": make_l,
        "model": model_l,
        "make_model": f"{make_l}__{model_l}" if make_l and model_l else None,
        "year": float(year_val) if pd.notna(year_val) else None,
        "mileage": float(mileage_val) if pd.notna(mileage_val) else None,
        "car_age": car_age,
        "miles_per_year": miles_per_year,
        "mileage_bucket": pp.mileage_bucket(mileage_val if pd.notna(mileage_val) else None),
        "is_luxury_make": int(make_l in pp.LUXURY_MAKES),
        "is_new": int(bool(is_new)),
        "trim": _norm(trim),
        "body_style": _norm(body_style),
        "drivetrain": _norm(drivetrain),
        "transmission_type": _norm(transmission_type),
        "fuel_type_group": _norm(fuel_type_group),
        "car_color": _norm(car_color),
        "title_status": _norm(title_status),
        "accident_history": _norm(accident_history),
        "engine_size_l": pd.to_numeric(engine_size_l, errors="coerce") if engine_size_l not in (None, "") else None,
        "mpg": pd.to_numeric(mpg, errors="coerce") if mpg not in (None, "") else None,
        "owner_count": pd.to_numeric(owner_count, errors="coerce") if owner_count not in (None, "") else None,
        "cpo": int(bool(cpo)) if cpo is not None else None,
        "region": _norm(region) or "unknown",
    }


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
        "conformal_delta": conformal.load_delta(ARTIFACT_DIR),
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
    delta = cache["conformal_delta"]
    x = _row_from_listing(listing, cache["feature_names"], cache["cat_features"])

    # q10/q90 get the conformal (CQR) widening in log space; q50/mean are as-is.
    lo = float(np.expm1(models["q10"].predict(x)[0] - delta))
    hi = float(np.expm1(models["q90"].predict(x)[0] + delta))
    median_raw = float(np.expm1(models["q50"].predict(x)[0]))
    point = float(np.expm1(models["mean"].predict(x)[0]))

    # Enforce a non-crossing band: lo <= median <= hi even if the independently
    # trained quantile models disagree for this particular input.
    band_lo, median, band_hi = sorted([lo, median_raw, hi])

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
