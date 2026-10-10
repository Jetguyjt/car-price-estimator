"""
Central path configuration for the whole project.

Every script imports its input/output locations from here instead of hardcoding
relative strings. Paths are resolved from the project root (the parent of this
`src/` directory), so scripts run correctly no matter what the current working
directory is:

    python3 src/train_price_model.py          # from repo root
    cd src && python3 train_price_model.py     # from src/
    python3 -m pytest tests/                   # from anywhere

Data layout (raw -> interim -> processed):

    data/raw/        immutable scraper output
    data/interim/    cleaned raw (output of clean_raw_data.py)
    data/processed/  model-ready table (output of preprocess_cars_data.py)
    data/snapshots/  optional frozen point-in-time CSVs
"""

from pathlib import Path

# --- roots ---
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
SNAPSHOT_DIR = DATA_DIR / "snapshots"

ARTIFACT_DIR = PROJECT_ROOT / "model_artifacts"
EVAL_DIR = ARTIFACT_DIR / "eval"
LOG_DIR = PROJECT_ROOT / "logs"

# --- canonical files ---
RAW_CSV = RAW_DIR / "vehicle_listings_raw.csv"
CLEAN_CSV = INTERIM_DIR / "vehicle_listings_clean.csv"
PROCESSED_CSV = PROCESSED_DIR / "vehicle_listings_processed.csv"


def ensure_output_dirs() -> None:
    """Create the directories that scripts write into (idempotent)."""
    for d in (INTERIM_DIR, PROCESSED_DIR, ARTIFACT_DIR, EVAL_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)


# Make output dirs exist on import so downstream scripts never trip on a missing
# folder. This is cheap and idempotent.
ensure_output_dirs()
