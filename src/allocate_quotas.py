"""
Allocate per-ZIP3 collection quotas and seed the scrape_state work queue.

Quota logic (per plan): quota[zip3] = clip(TARGET_TOTAL * pop_share, MIN_Q, MAX_Q).
Dense metros cap at MAX_Q so no single market dominates the dataset; rural ZIP3s
floor at MIN_Q (they'll usually exhaust their inventory before hitting it).

Each ZIP3 gets work units = ZIP3_BANDS x STOCK_TYPES for the given site, seeded
at the ZIP3's most-populous member zip (seed_zip5) with a fixed search radius.
Luxury inventory is too sparse per-ZIP3, so bands >= $80k also get NATIONWIDE
sweep units (zip3 '000') - that's where most six-figure rows will come from.

Run:
    python3 src/allocate_quotas.py --site cars_com          # seed/refresh units
    python3 src/allocate_quotas.py --site cars_com --show   # just print the plan
"""

import argparse
from pathlib import Path

import pandas as pd

import paths
import store

ZIP3_REFERENCE_CSV = Path(paths.DATA_DIR) / "reference" / "zip3_reference.csv"

TARGET_TOTAL = 175_000
MIN_QUOTA = 40
MAX_QUOTA = 1_500

# Coarser than the old nationwide 25-band split: per-ZIP3 searches are small, so
# 8 bands is enough to work within pagination limits while keeping the unit count sane.
ZIP3_BANDS = [
    (0, 5_000), (5_000, 10_000), (10_000, 15_000), (15_000, 20_000),
    (20_000, 30_000), (30_000, 45_000), (45_000, 80_000), (80_000, None),
]
NATIONWIDE_LUXURY_BANDS = [(80_000, 120_000), (120_000, 200_000), (200_000, 500_000)]
NATIONWIDE_ZIP3 = "000"

STOCK_TYPES = ["used", "new"]


def load_reference() -> pd.DataFrame:
    ref = pd.read_csv(ZIP3_REFERENCE_CSV, dtype={"zip3": str, "seed_zip5": str})
    ref["zip3"] = ref["zip3"].str.zfill(3)
    ref["seed_zip5"] = ref["seed_zip5"].str.zfill(5)
    return ref


def compute_quotas(ref: pd.DataFrame, target_total: int = TARGET_TOTAL) -> pd.DataFrame:
    out = ref.copy()
    share = out["population"] / out["population"].sum()
    out["quota"] = (target_total * share).round().clip(MIN_QUOTA, MAX_QUOTA).astype(int)
    return out


def build_units(quotas: pd.DataFrame, site: str) -> list:
    units = []
    for _, r in quotas.iterrows():
        for lo, hi in ZIP3_BANDS:
            for stock in STOCK_TYPES:
                units.append({
                    "site": site, "zip3": r["zip3"], "band_min": lo,
                    "band_max": hi, "stock_type": stock, "quota": int(r["quota"]),
                })
    for lo, hi in NATIONWIDE_LUXURY_BANDS:
        units.append({
            "site": site, "zip3": NATIONWIDE_ZIP3, "band_min": lo,
            "band_max": hi, "stock_type": "used", "quota": None,
        })
    return units


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="cars_com")
    ap.add_argument("--target", type=int, default=TARGET_TOTAL)
    ap.add_argument("--show", action="store_true", help="print plan without writing")
    args = ap.parse_args()

    ref = load_reference()
    quotas = compute_quotas(ref, args.target)
    total = quotas["quota"].sum()
    print(f"{len(quotas)} zip3s | planned quota sum: {total:,} "
          f"(target {args.target:,}; clip [{MIN_QUOTA}, {MAX_QUOTA}])")
    print(quotas.nlargest(5, "quota")[["zip3", "seed_zip5", "population", "quota"]]
          .to_string(index=False))

    if args.show:
        return
    units = build_units(quotas, args.site)
    conn = store.connect()
    added = store.seed_work_units(conn, units)
    print(f"seeded {added} new work units for {args.site} "
          f"({len(units) - added} already existed)")


if __name__ == "__main__":
    main()
