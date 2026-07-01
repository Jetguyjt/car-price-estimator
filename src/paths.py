"""
Central path configuration for the whole project.

Every script imports its input/output locations from here instead of hardcoding
relative strings. Paths are resolved from the project root (the parent of this
`src/` directory), so scripts run correctly no matter what the current working
directory is:

    python3 src/train_price_model.py          # from repo root
    cd src && python3 train_price_model.py     # from src/
    python3 -m pytest tests/                   # from anywhere

Data layout (cookiecutter-style raw -> interim -> processed):

    data/raw/        immutable scraper output
    data/interim/    cleaned raw (output of clean_raw_data.py)
    data/processed/  model-ready table (output of preprocess_cars_data.py)
    data/snapshots/  frozen point-in-time CSVs used for docs/tests
    data/benchmarks/ external benchmark templates (KBB/Edmunds)
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
BENCHMARK_DIR = DATA_DIR / "benchmarks"

ARTIFACT_DIR = PROJECT_ROOT / "model_artifacts"
EVAL_DIR = ARTIFACT_DIR / "eval"
LOG_DIR = PROJECT_ROOT / "logs"

# --- canonical files ---
RAW_CSV = RAW_DIR / "cars_com_raw.csv"                 # scraper output
CLEAN_CSV = INTERIM_DIR / "cars_com_clean.csv"         # clean_raw_data.py output
PROCESSED_CSV = PROCESSED_DIR / "cars_com_processed.csv"  # preprocess output
BENCHMARK_CSV = BENCHMARK_DIR / "benchmark_kbb.csv"
SCRAPER_STATE = PROJECT_ROOT / "scraper_state.json"    # per-partition resume checkpoint


def ensure_output_dirs() -> None:
    """Create the directories that scripts write into (idempotent)."""
    for d in (INTERIM_DIR, PROCESSED_DIR, ARTIFACT_DIR, EVAL_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)


# Make output dirs exist on import so downstream scripts never trip on a missing
# folder. This is cheap and idempotent.
ensure_output_dirs()
