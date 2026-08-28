"""
SQLite persistence for multi-site, ZIP3-partitioned collection.

Why a database now: the CSV appender was fine for one site and 13k rows, but the
expansion means ~150-200k rows, several sites writing the same schema, VIN-keyed
cross-site dedup, and collection running on a a different collection machine than
training. A single SQLite file gives atomic upserts, indexes, and a mergeable
shard format - while `export_training_csv.py` keeps the downstream
clean -> preprocess -> train pipeline working on a plain CSV, unchanged.

Tables
------
listings      one row per (site, listing_key); card fields + detail fields.
scrape_state  one row per work unit (site, zip3, price band, stock type):
              quota, last_page, status - the generalization of the old
              scraper_state.json partition checkpoints.
vin_specs     cache of NHTSA vPIC decode results, one row per VIN.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import paths

DB_PATH = Path(paths.DATA_DIR) / "listings.db"

# Card + detail fields shared with the earlier CSV schema,
# minus enriched_at which is a real column below.
LEGACY_COLUMNS = [
    "zip", "region", "scraped_at", "listing_url",
    "title", "year", "make", "model", "trim", "price", "mileage",
    "body_style", "stock_type", "stock_number",
    "exterior_color", "interior_color", "fuel_type_raw", "drivetrain_raw",
    "transmission_raw", "engine_raw", "mpg_text", "mpg_city", "mpg_highway",
    "mpg_combined", "mpg_equivalent",
    "title_status_raw", "no_accidents_raw", "owner_count", "cpo_raw",
]

# New columns for the multi-site / regional expansion.
NEW_COLUMNS = [
    "vin", "seed_zip3", "dealer_city", "dealer_state",
    "listing_type",       # dealer | private | auction
    "first_seen", "last_seen", "enriched_at",
]

LISTING_COLUMNS = ["site", "listing_key"] + LEGACY_COLUMNS + NEW_COLUMNS

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS listings (
    site         TEXT NOT NULL,
    listing_key  TEXT NOT NULL,
    {", ".join(f"{c} TEXT" for c in LEGACY_COLUMNS + NEW_COLUMNS)},
    PRIMARY KEY (site, listing_key)
);
CREATE INDEX IF NOT EXISTS idx_listings_vin ON listings(vin) WHERE vin IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_listings_seed_zip3 ON listings(seed_zip3);
CREATE INDEX IF NOT EXISTS idx_listings_enrich ON listings(site, enriched_at);

CREATE TABLE IF NOT EXISTS scrape_state (
    site        TEXT NOT NULL,
    zip3        TEXT NOT NULL,     -- '000' for nationwide luxury sweeps
    band_min    INTEGER,
    band_max    INTEGER,           -- NULL = open-ended
    stock_type  TEXT NOT NULL,
    quota       INTEGER,           -- rows wanted for this zip3 (denormalized)
    collected   INTEGER DEFAULT 0, -- new rows this unit has produced
    last_page   INTEGER DEFAULT 0,
    status      TEXT DEFAULT 'pending',  -- pending|in_progress|done|exhausted
    updated_at  TEXT,
    PRIMARY KEY (site, zip3, band_min, stock_type)
);

CREATE TABLE IF NOT EXISTS vin_specs (
    vin        TEXT PRIMARY KEY,
    decoded_at TEXT,
    payload    TEXT               -- raw vPIC JSON for the fields we keep
);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    # WAL lets the enricher read while the collector writes.
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


# ---------------------------- listings ----------------------------

def upsert_listings(conn: sqlite3.Connection, rows: Iterable[Dict]) -> int:
    """Insert new listings or refresh existing ones.

    A re-seen listing updates last_seen and any newly non-null fields, but never
    overwrites an existing non-null value with null (a card re-scrape must not
    erase detail-page enrichment).
    """
    now = utcnow()
    inserted = 0
    cols = LISTING_COLUMNS
    placeholders = ", ".join("?" for _ in cols)
    update_cols = [c for c in cols if c not in ("site", "listing_key", "first_seen")]
    update_clause = ", ".join(
        f"{c} = COALESCE(excluded.{c}, {c})" for c in update_cols if c != "last_seen"
    ) + ", last_seen = excluded.last_seen"
    sql = (
        f"INSERT INTO listings ({', '.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT(site, listing_key) DO UPDATE SET {update_clause}"
    )
    for row in rows:
        values = []
        for c in cols:
            v = row.get(c)
            if v is None or v == "":
                v = None
            values.append(v if v is None else str(v))
        idx_first = cols.index("first_seen")
        idx_last = cols.index("last_seen")
        values[idx_first] = values[idx_first] or now
        values[idx_last] = now
        cur = conn.execute(sql, values)
        inserted += cur.rowcount
    conn.commit()
    return inserted


def existing_keys(conn: sqlite3.Connection, site: str) -> set:
    return {r["listing_key"] for r in conn.execute(
        "SELECT listing_key FROM listings WHERE site = ?", (site,))}


def rows_for_zip3(conn: sqlite3.Connection, zip3: str) -> int:
    (n,) = conn.execute(
        "SELECT COUNT(*) FROM listings WHERE seed_zip3 = ?", (zip3,)).fetchone()
    return n


def unenriched(conn: sqlite3.Connection, site: Optional[str] = None,
               limit: Optional[int] = None) -> List[sqlite3.Row]:
    """Rows still needing the detail-page pass, luxury-first (see Phase C plan)."""
    q = (
        "SELECT site, listing_key, listing_url, price FROM listings "
        "WHERE enriched_at IS NULL AND listing_url IS NOT NULL "
    )
    args: list = []
    if site:
        q += "AND site = ? "
        args.append(site)
    q += "ORDER BY CAST(price AS REAL) DESC "
    if limit:
        q += "LIMIT ?"
        args.append(limit)
    return list(conn.execute(q, args))


def mark_enriched(conn: sqlite3.Connection, site: str, listing_key: str,
                  detail_fields: Dict) -> None:
    detail_fields = dict(detail_fields)
    detail_fields["enriched_at"] = utcnow()
    sets = ", ".join(f"{c} = ?" for c in detail_fields)
    conn.execute(
        f"UPDATE listings SET {sets} WHERE site = ? AND listing_key = ?",
        list(detail_fields.values()) + [site, listing_key],
    )
    conn.commit()


# ---------------------------- work units ----------------------------

def seed_work_units(conn: sqlite3.Connection, units: Iterable[Dict]) -> int:
    """Insert work units if absent; existing units keep their progress."""
    sql = (
        "INSERT OR IGNORE INTO scrape_state "
        "(site, zip3, band_min, band_max, stock_type, quota, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)"
    )
    n = 0
    now = utcnow()
    for u in units:
        cur = conn.execute(sql, (u["site"], u["zip3"], u.get("band_min"),
                                 u.get("band_max"), u["stock_type"],
                                 u.get("quota"), now))
        n += cur.rowcount
    conn.commit()
    return n


def claim_next_unit(conn: sqlite3.Connection, site: str) -> Optional[sqlite3.Row]:
    """Return the next pending or in-progress unit for a site."""
    return conn.execute(
        "SELECT * FROM scrape_state WHERE site = ? AND status IN "
        "('pending', 'in_progress') ORDER BY status DESC, zip3, band_min LIMIT 1",
        (site,),
    ).fetchone()


def update_unit(conn: sqlite3.Connection, site: str, zip3: str, band_min,
                stock_type: str, *, last_page: Optional[int] = None,
                status: Optional[str] = None, collected_delta: int = 0) -> None:
    sets, args = ["updated_at = ?"], [utcnow()]
    if last_page is not None:
        sets.append("last_page = ?")
        args.append(last_page)
    if status is not None:
        sets.append("status = ?")
        args.append(status)
    if collected_delta:
        sets.append("collected = collected + ?")
        args.append(collected_delta)
    args += [site, zip3, band_min, stock_type]
    conn.execute(
        f"UPDATE scrape_state SET {', '.join(sets)} "
        "WHERE site = ? AND zip3 = ? AND band_min IS ? AND stock_type = ?", args)
    conn.commit()
