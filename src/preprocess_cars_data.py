"""
Preprocess raw Cars.com scrape into a model-ready table.

Changes vs the old version:
- No more -999999 / __missing_sentinel__ shenanigans. NaN is NaN.
- Wider price clip so luxury cars survive (they're real listings).
- Region kept as a feature.
- Drops only obvious garbage; lets the model handle nuance.
"""

import re
from datetime import datetime
from typing import Optional

import pandas as pd

import paths


# By default preprocess consumes the CLEANED table (output of clean_raw_data.py),
# not the raw scrape. Pass an explicit input path to override.
RAW_INPUT_CSV_PATH = str(paths.CLEAN_CSV)
OUTPUT_CSV_PATH = str(paths.PROCESSED_CSV)
VERBOSE = True

# Price clipping is a tradeoff: too tight and you lose luxury, too loose and
# you keep typo'd listings. 0.005/0.995 keeps the genuine $80k–$150k cars in.
REMOVE_PRICE_OUTLIERS = True
PRICE_LOWER_Q = 0.005
PRICE_UPPER_Q = 0.995

REMOVE_PRICE_PER_MILE_OUTLIERS = True
PRICE_PER_MILE_LOWER_Q = 0.01
PRICE_PER_MILE_UPPER_Q = 0.99

MAX_MILEAGE_ALLOWED = 350000

# Rare-category grouping: keeps high-cardinality columns from blowing up,
# and gives the model a sensible bucket for unseen values at predict time.
GROUP_RARE_CATEGORIES = True
MIN_TRIM_COUNT = 10
MIN_BODY_STYLE_COUNT = 10
MIN_COLOR_COUNT = 10


def log(message: str) -> None:
    if VERBOSE:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


# ------------------------------ normalizers ------------------------------

def normalize_color(value: Optional[str]) -> Optional[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = re.sub(r"[^a-z\s]", " ", str(value).lower())
    text = " ".join(text.split())
    if not text:
        return None

    if any(k in text for k in ["black", "onyx", "ebony", "jet"]):
        return "black"
    if any(k in text for k in ["white", "ivory", "snow", "pearl"]):
        return "white"
    if any(k in text for k in ["gray", "grey", "charcoal", "graphite", "gunmetal"]):
        return "gray"
    if any(k in text for k in ["silver", "aluminum"]):
        return "silver"
    if any(k in text for k in ["blue", "navy", "indigo", "cobalt"]):
        return "blue"
    if any(k in text for k in ["red", "burgundy", "maroon", "crimson"]):
        return "red"
    if any(k in text for k in ["green", "olive"]):
        return "green"
    if any(k in text for k in ["brown", "beige", "tan", "mocha", "bronze", "champagne", "gold"]):
        return "brown"
    if any(k in text for k in ["orange", "copper"]):
        return "orange"
    if "yellow" in text:
        return "yellow"
    if "purple" in text or "violet" in text:
        return "purple"
    return "other"


def normalize_drivetrain(value) -> Optional[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = re.sub(r"[^A-Z0-9]+", " ", str(value).upper()).strip()
    if not text:
        return None
    if "AWD" in text or "ALL WHEEL" in text:
        return "awd"
    if "FWD" in text or "FRONT WHEEL" in text:
        return "fwd"
    if "RWD" in text or "REAR WHEEL" in text:
        return "rwd"
    if "4WD" in text or "4X4" in text or "FOUR WHEEL" in text:
        return "4wd"
    return None


def normalize_transmission(value) -> Optional[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).lower()
    if not text.strip():
        return None
    if "cvt" in text:
        return "cvt"
    if "manual" in text or " mt" in text:
        return "manual"
    if "auto" in text or "automatic" in text:
        return "automatic"
    return None


def normalize_fuel(value) -> Optional[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).lower()
    if not text.strip():
        return None
    if "electric" in text or text.strip() == "ev":
        return "ev"
    if "hybrid" in text:
        return "hybrid"
    if "diesel" in text:
        return "diesel"
    if "gas" in text or "unleaded" in text:
        return "gas"
    return None


def normalize_title_status(value) -> Optional[str]:
    if isinstance(value, bool):
        return "clean" if value else None
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).lower()
    if "salvage" in text:
        return "salvage"
    if "rebuilt" in text:
        return "rebuilt"
    if "clean" in text or text == "true":
        return "clean"
    return None


def normalize_accident(value) -> Optional[str]:
    if isinstance(value, bool):
        return "no_accident" if value else "had_accident"
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).lower()
    if "no" in text and "accident" in text:
        return "no_accident"
    if "accident" in text:
        return "had_accident"
    if text == "true":
        return "no_accident"
    if text == "false":
        return "had_accident"
    return None


def parse_engine_size_l(text) -> Optional[float]:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*[lL]", str(text))
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def parse_mpg_from_text(text) -> dict:
    out = {"mpg_city": None, "mpg_highway": None, "mpg_combined": None, "mpg_equivalent": None}
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return out
    lower = str(text).lower()
    nums = [int(n) for n in re.findall(r"\d+", lower)]
    if "mpge" in lower and nums:
        out["mpg_equivalent"] = nums[0]
    elif len(nums) >= 2:
        out["mpg_city"] = nums[0]
        out["mpg_highway"] = nums[1]
    elif len(nums) == 1:
        out["mpg_combined"] = nums[0]
    return out


def group_rare(series: pd.Series, min_count: int) -> pd.Series:
    counts = series.value_counts(dropna=False)
    rare_values = set(counts[counts < min_count].index.tolist())
    return series.apply(lambda v: "__rare__" if v in rare_values else v)


# ------------------------------ main pipeline ------------------------------

def preprocess_for_model(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    numeric_cols = [
        "price", "year", "mileage", "owner_count", "engine_size_l",
        "mpg_city", "mpg_highway", "mpg_combined", "mpg_equivalent",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    current_year = datetime.now().year

    # --- core required fields ---
    for col in ["make", "model"]:
        if col not in out.columns:
            out[col] = pd.NA
        out[col] = out[col].astype("string").str.strip()

    before = len(out)
    out = out.dropna(subset=["price", "year", "mileage", "make", "model"])
    out = out[(out["make"] != "") & (out["model"] != "")]
    out = out[out["price"] > 500]                       # under $500 = junk listing
    out = out[out["mileage"].between(0, MAX_MILEAGE_ALLOWED)]
    out = out[out["year"].between(1990, current_year + 1)]
    log(f"After core-field filter: {len(out)} (dropped {before - len(out)})")

    # --- normalizations ---
    out["make"] = out["make"].astype(str).str.lower().str.strip()
    out["model"] = out["model"].astype(str).str.lower().str.strip()
    out["trim"] = out.get("trim").astype("string").str.lower().str.strip() if "trim" in out.columns else pd.NA
    out["body_style"] = out.get("body_style").astype("string").str.lower().str.strip() if "body_style" in out.columns else pd.NA

    # New cars have 0 prior owners by definition
    if "stock_type" in out.columns:
        stock_type = out["stock_type"].astype("string").str.lower().fillna("")
        if "owner_count" not in out.columns:
            out["owner_count"] = pd.NA
        out["owner_count"] = pd.to_numeric(out["owner_count"], errors="coerce")
        out.loc[stock_type == "new", "owner_count"] = 0

    # CPO flag
    cpo_source = out.get("cpo_raw", out.get("cpo", pd.Series(index=out.index, dtype="object")))
    out["cpo"] = cpo_source.map({True: 1, False: 0, "true": 1, "false": 0, "True": 1, "False": 0})
    out["cpo"] = pd.to_numeric(out["cpo"], errors="coerce")

    # Drivetrain / fuel / transmission / engine
    out["drivetrain"] = out.get("drivetrain_raw").apply(normalize_drivetrain) if "drivetrain_raw" in out.columns else pd.NA
    out["fuel_type_group"] = out.get("fuel_type_raw").apply(normalize_fuel) if "fuel_type_raw" in out.columns else pd.NA
    out["transmission_type"] = out.get("transmission_raw").apply(normalize_transmission) if "transmission_raw" in out.columns else pd.NA

    if "engine_raw" in out.columns:
        engine_size_from_text = out["engine_raw"].apply(parse_engine_size_l)
        out["engine_size_l"] = out.get("engine_size_l", pd.Series(index=out.index, dtype="float64"))
        out["engine_size_l"] = pd.to_numeric(out["engine_size_l"], errors="coerce").fillna(engine_size_from_text)

    # MPG
    if "mpg_text" in out.columns:
        parsed = out["mpg_text"].apply(parse_mpg_from_text).apply(pd.Series)
        for col in ["mpg_city", "mpg_highway", "mpg_combined", "mpg_equivalent"]:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce").fillna(parsed[col])
            else:
                out[col] = parsed[col]

    # Title / accident
    out["title_status"] = out.get("title_status_raw").apply(normalize_title_status) if "title_status_raw" in out.columns else pd.NA
    out["accident_history"] = out.get("no_accidents_raw").apply(normalize_accident) if "no_accidents_raw" in out.columns else pd.NA

    # Color
    color_source = out.get("exterior_color", out.get("car_color", pd.Series(index=out.index, dtype="object")))
    out["car_color"] = color_source.apply(normalize_color)

    # Region (from scraper)
    if "region" not in out.columns:
        out["region"] = "unknown"
    out["region"] = out["region"].astype("string").str.strip().fillna("unknown")
    if "zip" in out.columns:
        out["zip"] = out["zip"].astype("string")

    # --- engineered features ---
    out["make_model"] = out["make"] + "__" + out["model"]
    out["car_age"] = (current_year - out["year"]).clip(lower=0)
    age_safe = out["car_age"].replace(0, 1)
    out["miles_per_year"] = (out["mileage"] / age_safe).replace([float("inf"), -float("inf")], pd.NA)

    out["price_per_mile"] = pd.Series(float("nan"), index=out.index, dtype="float64")
    has_mileage = out["mileage"] > 0
    out.loc[has_mileage, "price_per_mile"] = out.loc[has_mileage, "price"] / out.loc[has_mileage, "mileage"]

    # Single MPG number — average of city/hwy when both present, else fall back
    out["mpg"] = pd.to_numeric(out.get("mpg_city"), errors="coerce")
    if "mpg_highway" in out.columns:
        both = out["mpg"].notna() & out["mpg_highway"].notna()
        out.loc[both, "mpg"] = (out.loc[both, "mpg"] + pd.to_numeric(out.loc[both, "mpg_highway"], errors="coerce")) / 2.0
        out["mpg"] = out["mpg"].fillna(pd.to_numeric(out["mpg_highway"], errors="coerce"))
    if "mpg_combined" in out.columns:
        out["mpg"] = out["mpg"].fillna(pd.to_numeric(out["mpg_combined"], errors="coerce"))
    if "mpg_equivalent" in out.columns:
        out["mpg"] = out["mpg"].fillna(pd.to_numeric(out["mpg_equivalent"], errors="coerce"))

    out["mileage_bucket"] = pd.cut(
        out["mileage"],
        bins=[-1, 15000, 60000, 120000, float("inf")],
        labels=["very_low", "low", "medium", "high"],
    ).astype("string")

    luxury_makes = {
        "acura", "alfa romeo", "aston martin", "audi", "bentley", "bmw", "cadillac",
        "genesis", "infiniti", "jaguar", "land rover", "lexus", "lincoln", "lucid",
        "maserati", "mercedes-benz", "mercedes", "porsche", "rolls-royce", "volvo",
        "ferrari", "lamborghini", "mclaren", "rivian", "tesla",
    }
    out["is_luxury_make"] = out["make"].isin(luxury_makes).astype(int)

    out["is_new"] = (out.get("stock_type", pd.Series(index=out.index, dtype="object")).astype(str).str.lower() == "new").astype(int)

    # --- outlier filtering ---
    if REMOVE_PRICE_OUTLIERS and len(out) > 50:
        low = out["price"].quantile(PRICE_LOWER_Q)
        high = out["price"].quantile(PRICE_UPPER_Q)
        before = len(out)
        out = out[out["price"].between(low, high)]
        log(f"Price outlier filter [{low:.0f}, {high:.0f}] removed {before - len(out)} rows.")

    if REMOVE_PRICE_PER_MILE_OUTLIERS and len(out) > 50:
        ppm_all = pd.to_numeric(out["price_per_mile"], errors="coerce")
        ppm = ppm_all.dropna()
        if len(ppm) > 50:
            low = ppm.quantile(PRICE_PER_MILE_LOWER_Q)
            high = ppm.quantile(PRICE_PER_MILE_UPPER_Q)
            before = len(out)
            keep = ppm_all.isna() | ppm_all.between(low, high)
            out = out[keep]
            log(f"Price-per-mile outlier filter removed {before - len(out)} rows.")

    if GROUP_RARE_CATEGORIES:
        out["trim"] = group_rare(out["trim"].fillna("__unknown__"), MIN_TRIM_COUNT)
        out["body_style"] = group_rare(out["body_style"].fillna("__unknown__"), MIN_BODY_STYLE_COUNT)
        out["car_color"] = group_rare(out["car_color"].fillna("__unknown__"), MIN_COLOR_COUNT)

    # final column order — keep NaN, no sentinels
    ordered_cols = [
        "listing_id", "listing_key", "zip", "region",
        "price", "make", "model", "make_model", "year", "mileage", "car_age",
        "miles_per_year", "price_per_mile", "mileage_bucket", "is_luxury_make", "is_new",
        "trim", "title_status", "accident_history", "owner_count", "cpo",
        "drivetrain", "engine_size_l", "transmission_type", "fuel_type_group",
        "mpg", "car_color", "body_style",
    ]
    if "listing_id" not in out.columns:
        out.insert(0, "listing_id", range(len(out)))
    out = out[[c for c in ordered_cols if c in out.columns]].reset_index(drop=True)
    out["listing_id"] = range(len(out))
    return out


def main() -> None:
    # Optional CLI overrides: python3 preprocess_cars_data.py [input.csv] [output.csv]
    import sys
    input_path = sys.argv[1] if len(sys.argv) > 1 else RAW_INPUT_CSV_PATH
    output_path = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_CSV_PATH

    log(f"Reading raw data: {input_path}")
    raw_df = pd.read_csv(input_path)
    if raw_df.empty:
        log("Raw CSV is empty.")
        return

    processed_df = preprocess_for_model(raw_df)
    processed_df.to_csv(output_path, index=False)
    log(f"Saved processed data: {output_path}")
    log(f"Rows: raw={len(raw_df)} processed={len(processed_df)}")
    print(processed_df.head(5).to_string(index=False))

    # quick non-null summary so you can see how much detail-page data survived
    print("\nNon-null counts (top fields):")
    for col in ["drivetrain", "fuel_type_group", "transmission_type", "engine_size_l",
                "mpg", "car_color", "owner_count", "cpo", "title_status", "accident_history"]:
        if col in processed_df.columns:
            n = int(processed_df[col].notna().sum())
            pct = 100 * n / len(processed_df) if len(processed_df) else 0
            print(f"  {col:22s} {n:>6d} ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
