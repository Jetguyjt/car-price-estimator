"""
Merge a listings.db shard (e.g. scraped on another machine) into the local database.

Same row-level semantics as store.upsert_listings: new (site, listing_key) rows
insert; existing rows take the shard's non-null fields but never lose a non-null
value to a null. scrape_state merges by taking the furthest progress per unit.

Run:
    python3 src/db_merge.py /path/to/shard.db
"""

import sys
from pathlib import Path

import store


def merge(shard_path: Path, into_path: Path = None) -> dict:
    src = store.connect(shard_path)
    dst = store.connect(into_path)

    rows = [dict(r) for r in src.execute("SELECT * FROM listings")]
    before = dst.execute("SELECT COUNT(*) FROM listings").fetchone()[0]
    # upsert_listings stamps last_seen=now; preserve the shard's own timestamps.
    cols = store.LISTING_COLUMNS
    placeholders = ", ".join("?" for _ in cols)
    update_cols = [c for c in cols if c not in ("site", "listing_key")]
    update_clause = ", ".join(f"{c} = COALESCE(excluded.{c}, {c})" for c in update_cols)
    sql = (
        f"INSERT INTO listings ({', '.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT(site, listing_key) DO UPDATE SET {update_clause}"
    )
    for row in rows:
        dst.execute(sql, [row.get(c) for c in cols])

    for r in src.execute("SELECT * FROM scrape_state"):
        dst.execute(
            "INSERT INTO scrape_state (site, zip3, band_min, band_max, stock_type, "
            "quota, collected, last_page, status, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(site, zip3, band_min, stock_type) DO UPDATE SET "
            "collected = MAX(collected, excluded.collected), "
            "last_page = MAX(last_page, excluded.last_page), "
            "status = CASE WHEN excluded.last_page > last_page "
            "         THEN excluded.status ELSE status END, "
            "updated_at = MAX(updated_at, excluded.updated_at)",
            (r["site"], r["zip3"], r["band_min"], r["band_max"], r["stock_type"],
             r["quota"], r["collected"], r["last_page"], r["status"], r["updated_at"]),
        )
    for r in src.execute("SELECT * FROM vin_specs"):
        dst.execute(
            "INSERT OR IGNORE INTO vin_specs (vin, decoded_at, payload) VALUES (?, ?, ?)",
            (r["vin"], r["decoded_at"], r["payload"]),
        )
    dst.commit()
    after = dst.execute("SELECT COUNT(*) FROM listings").fetchone()[0]
    return {"shard_rows": len(rows), "new_rows": after - before, "total_rows": after}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python3 src/db_merge.py <shard.db> [into.db]")
    into = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    stats = merge(Path(sys.argv[1]), into)
    print(f"merged {stats['shard_rows']} shard rows -> +{stats['new_rows']} new "
          f"(total {stats['total_rows']})")
