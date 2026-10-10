"""
Clean raw vehicle listings into a trustworthy intermediate table.

This is the *hygiene* stage of the pipeline and it sits deliberately between the
scraper and the feature-engineering step:

    data/raw/vehicle_listings_raw.csv          (collector output)
        -> clean_raw_data.py ->
    data/interim/vehicle_listings_clean.csv    (this file's output)
        -> preprocess_cars_data.py ->
    data/processed/vehicle_listings_processed.csv

Why a separate stage instead of folding it into preprocessing? Because these two
jobs have different contracts. Cleaning removes rows that are *wrong* (missing
required fields, impossible values, scrape artifacts) and keeps the raw schema
intact. Preprocessing then *transforms* what survives (normalization, feature
engineering, rare-category grouping). Keeping them apart means the cleaning rules
are auditable on their own and the intermediate CSV is still human-readable.

The rule that earns this stage its keep is the check for payment-amount artifacts.
Some listing cards can surface a monthly or down-payment figure where the full price
should be. A flat minimum-price rule does not catch every case, so a second rule
flags nearly new, low-mileage vehicles with implausibly low captured prices.

Run:
    python3 src/clean_raw_data.py                 # raw -> interim (default paths)
    python3 src/clean_raw_data.py in.csv out.csv  # explicit override
"""

import sys
from datetime import datetime

import pandas as pd

import paths


VERBOSE = True

# --- impossible / junk thresholds ---
MIN_PRICE = 500          # at or below this, a "price" is a placeholder, not a car
MAX_MILEAGE = 350_000    # beyond this, almost always a data-entry error
MIN_YEAR = 1990          # older than this is out of scope for this model
# MAX_YEAR is current_year + 1 (next-model-year cars are legitimately listed)

# --- payment-amount artifact rule ---
# A nearly-new, low-mileage car listed for a few thousand dollars is not a real
# sale price; it is a captured monthly payment / down payment. All three
# conditions must hold, so genuinely cheap *old* high-mileage cars are untouched.
ARTIFACT_RECENT_WITHIN_YEARS = 2   # year >= current_year - this
ARTIFACT_MAX_MILEAGE = 30_000
ARTIFACT_MAX_PRICE = 12_000

REQUIRED_FIELDS = ["price", "year", "make", "model", "mileage"]
NUMERIC_FIELDS = ["price", "year", "mileage", "owner_count"]


def log(message: str) -> None:
    if VERBOSE:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def _recency_key(df: pd.DataFrame) -> pd.Series:
    """A sortable timestamp used to keep the freshest row per listing_key.

    Prefer detail-page enrichment time, fall back to scrape time. Unparseable /
    missing timestamps sort oldest so a row with a real timestamp always wins.
    """
    # utc=True normalizes the mix of tz-aware and tz-naive strings the scraper
    # emits into a single comparable, sortable dtype.
    missing = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    enriched_source = df["enriched_at"] if "enriched_at" in df.columns else missing
    scraped_source = df["scraped_at"] if "scraped_at" in df.columns else missing
    enriched = pd.to_datetime(enriched_source, errors="coerce", utc=True)
    scraped = pd.to_datetime(scraped_source, errors="coerce", utc=True)
    return enriched.fillna(scraped).fillna(pd.Timestamp.min.tz_localize("UTC"))


def clean_raw(
    df: pd.DataFrame,
    *,
    reference_year: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Return (clean_df, report). The report counts rows dropped per reason."""
    report: dict = {"rows_in": len(df)}
    out = df.copy()

    # Trim stray whitespace on text columns (keeps values otherwise raw).
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].astype("string").str.strip()
            # Treat blank strings as missing so required-field checks are honest.
            out[col] = out[col].replace({"": pd.NA})

    # Numeric coercion up front so every downstream comparison is numeric.
    for col in NUMERIC_FIELDS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    current_year = int(reference_year or datetime.now().year)
    max_year = current_year + 1

    # --- 1. exact duplicate rows ---
    before = len(out)
    out = out.drop_duplicates()
    report["dropped_exact_duplicate_rows"] = before - len(out)

    # --- 2. duplicate listing_key (keep freshest) ---
    if "listing_key" in out.columns:
        before = len(out)
        out = out.assign(_recency=_recency_key(out))
        out = out.sort_values("_recency")
        # Missing keys do not identify the same listing. Deduplicate only rows
        # with an actual key, otherwise every keyless row would collapse to one.
        populated_key = out["listing_key"].notna()
        duplicate_key = populated_key & out.duplicated(subset="listing_key", keep="last")
        out = out.loc[~duplicate_key].drop(columns="_recency")
        report["dropped_duplicate_listing_key"] = before - len(out)

    # --- 3. missing required fields ---
    before = len(out)
    out = out.dropna(subset=[c for c in REQUIRED_FIELDS if c in out.columns])
    report["dropped_missing_required_field"] = before - len(out)

    # --- 4. impossible / junk values ---
    before = len(out)
    out = out[out["price"] > MIN_PRICE]
    report["dropped_price_at_or_below_floor"] = before - len(out)

    before = len(out)
    out = out[out["mileage"].between(0, MAX_MILEAGE)]
    report["dropped_mileage_out_of_range"] = before - len(out)

    before = len(out)
    out = out[out["year"].between(MIN_YEAR, max_year)]
    report["dropped_year_out_of_range"] = before - len(out)

    # --- 5. payment-amount artifacts (the reason this stage exists) ---
    is_artifact = (
        (out["year"] >= current_year - ARTIFACT_RECENT_WITHIN_YEARS)
        & (out["mileage"] < ARTIFACT_MAX_MILEAGE)
        & (out["price"] < ARTIFACT_MAX_PRICE)
    )
    report["dropped_payment_amount_artifact"] = int(is_artifact.sum())
    # Keep a few examples so the report can show its work.
    report["artifact_examples"] = (
        out.loc[is_artifact, ["year", "make", "model", "mileage", "price"]]
        .sort_values("price")
        .head(6)
        .to_dict("records")
    )
    out = out[~is_artifact]

    report["rows_out"] = len(out)
    return out.reset_index(drop=True), report


def format_report(report: dict) -> str:
    lines = ["# Cleaning report", ""]
    lines.append(f"- Rows in (raw): **{report['rows_in']}**")
    lines.append(f"- Rows out (clean): **{report['rows_out']}**")
    total_dropped = report["rows_in"] - report["rows_out"]
    pct = 100 * total_dropped / report["rows_in"] if report["rows_in"] else 0
    lines.append(f"- Total dropped: **{total_dropped}** ({pct:.1f}%)")
    lines.append("")
    lines.append("| Reason | Rows dropped |")
    lines.append("|---|---|")
    reasons = [
        ("Exact duplicate rows", "dropped_exact_duplicate_rows"),
        ("Duplicate listing_key (kept freshest)", "dropped_duplicate_listing_key"),
        ("Missing required field", "dropped_missing_required_field"),
        (f"Price at/below ${MIN_PRICE} floor", "dropped_price_at_or_below_floor"),
        (f"Mileage outside 0-{MAX_MILEAGE:,}", "dropped_mileage_out_of_range"),
        (f"Year outside {MIN_YEAR}-next model year", "dropped_year_out_of_range"),
        ("Payment-amount artifact", "dropped_payment_amount_artifact"),
    ]
    for label, key in reasons:
        lines.append(f"| {label} | {report.get(key, 0)} |")
    if report.get("artifact_examples"):
        lines.append("")
        lines.append("### Payment-amount artifacts caught (examples)")
        lines.append("")
        lines.append("| Year | Make | Model | Mileage | Listed 'price' |")
        lines.append("|---|---|---|---|---|")
        for r in report["artifact_examples"]:
            lines.append(
                f"| {int(r['year'])} | {r['make']} | {r['model']} | "
                f"{int(r['mileage']):,} | ${int(r['price']):,} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    input_path = sys.argv[1] if len(sys.argv) > 1 else str(paths.RAW_CSV)
    output_path = sys.argv[2] if len(sys.argv) > 2 else str(paths.CLEAN_CSV)

    log(f"Reading raw data: {input_path}")
    raw_df = pd.read_csv(input_path)
    if raw_df.empty:
        raise SystemExit("Raw CSV is empty; refusing to leave an older cleaned file in place.")

    clean_df, report = clean_raw(raw_df)
    if clean_df.empty:
        raise SystemExit("Cleaning removed every row; no output was written.")
    clean_df.to_csv(output_path, index=False)
    log(f"Saved cleaned data: {output_path}")

    report_path = paths.INTERIM_DIR / "cleaning_report.md"
    report_md = format_report(report)
    report_path.write_text(report_md)
    log(f"Saved cleaning report: {report_path}")

    print()
    print(report_md)


if __name__ == "__main__":
    main()
