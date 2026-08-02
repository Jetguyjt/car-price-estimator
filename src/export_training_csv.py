"""
Export the SQLite listings store to a flat CSV for the training pipeline.

The clean -> preprocess -> train stages all consume a CSV; this keeps that
contract so the model pipeline is untouched by the storage change. The export
also carries the new expansion columns (site, vin, seed_zip3, dealer_city/state,
listing_type) which cleaning/preprocessing pick up when present.

Run:
    python3 src/export_training_csv.py            # -> data/raw/listings_export.csv
"""

from pathlib import Path

import pandas as pd

import paths
import store

EXPORT_CSV = Path(paths.RAW_DIR) / "listings_export.csv"


def export(db_path: Path = None, out_path: Path = EXPORT_CSV) -> pd.DataFrame:
    conn = store.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM listings", conn)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


if __name__ == "__main__":
    df = export()
    print(f"exported {len(df)} rows -> {EXPORT_CSV}")
    if len(df):
        print(df["site"].value_counts().to_string())
