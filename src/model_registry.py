"""
Lightweight model version registry.

`model_artifacts/` always holds the *current* model (so `predict_listing.py` and
`evaluate_model.py` never need to know about versions). This module snapshots each
trained model into `model_artifacts/archive/vN/` and records it in a registry, so
improving the model never destroys the one it replaced.

Layout:

    model_artifacts/
        lgbm_*.pkl, *.json, *.csv ...   <- current model (latest version)
        registry.json                   <- machine-readable version list
        MODEL_HISTORY.md                <- human-readable changelog
        archive/
            v1/ ...                      <- frozen copy of each past model
            v2/ ...

Typical use from training (after the new artifacts are written):

    import model_registry
    model_registry.register_current_model(ARTIFACT_DIR, metrics_payload=payload,
                                          notes="retrain on cleaned 2.5k dataset")

That call assigns the next version number, copies the current artifacts into
`archive/vN/`, and appends a row to the registry + changelog.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# The files that together constitute one model version. Missing ones are skipped,
# so this stays correct even if an older model predates a given artifact.
ARTIFACT_FILE_NAMES = [
    "lgbm_q10.pkl", "lgbm_q50.pkl", "lgbm_q90.pkl", "lgbm_mean.pkl",
    "feature_names.json", "cat_features.json", "category_levels.json",
    "category_counts.json", "feature_importance.csv", "test_predictions.csv",
    "metrics.json", "quick_model_writeup.md", "conformal.json",
    "conformal_mondrian.json", "make_model_price_enc.json",
]

REGISTRY_NAME = "registry.json"
HISTORY_NAME = "MODEL_HISTORY.md"
ARCHIVE_DIRNAME = "archive"


def _registry_path(artifact_dir: Path) -> Path:
    return artifact_dir / REGISTRY_NAME


def load_registry(artifact_dir: Path) -> Dict:
    path = _registry_path(artifact_dir)
    if path.exists():
        return json.loads(path.read_text())
    return {"latest_version": 0, "versions": []}


def save_registry(artifact_dir: Path, registry: Dict) -> None:
    _registry_path(artifact_dir).write_text(json.dumps(registry, indent=2))


def summarize_metrics(metrics_payload: Optional[Dict]) -> Dict:
    """Pull the headline numbers out of a metrics.json payload for the changelog."""
    if not metrics_payload:
        return {}
    tm = metrics_payload.get("test_metrics", {})
    ds = metrics_payload.get("dataset", {})
    return {
        "rows_total": ds.get("rows_total"),
        "rows_test": ds.get("rows_test"),
        "test_mae": tm.get("test_mae"),
        "test_r2": tm.get("test_r2"),
        "test_mape_percent": tm.get("test_mape_percent"),
        "interval_coverage_q10_q90": tm.get("interval_coverage_q10_q90"),
    }


def _copy_artifacts(artifact_dir: Path, dest: Path) -> List[str]:
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in ARTIFACT_FILE_NAMES:
        src = artifact_dir / name
        if src.exists():
            shutil.copy2(src, dest / name)
            copied.append(name)
    return copied


def register_current_model(
    artifact_dir,
    metrics_payload: Optional[Dict] = None,
    notes: str = "",
    created_at: Optional[str] = None,
) -> int:
    """Snapshot the model currently in `artifact_dir` as the next version.

    Reads metrics from `metrics_payload` if given, else from the on-disk
    `metrics.json`. Returns the assigned version number.
    """
    artifact_dir = Path(artifact_dir)
    registry = load_registry(artifact_dir)

    if metrics_payload is None:
        metrics_file = artifact_dir / "metrics.json"
        if metrics_file.exists():
            metrics_payload = json.loads(metrics_file.read_text())

    version = registry["latest_version"] + 1
    archive_dir = artifact_dir / ARCHIVE_DIRNAME / f"v{version}"
    copied = _copy_artifacts(artifact_dir, archive_dir)

    entry = {
        "version": version,
        "created_at": created_at or datetime.now().isoformat(timespec="seconds"),
        "notes": notes,
        "archive_path": str(archive_dir.relative_to(artifact_dir)),
        "artifacts": copied,
        "metrics": summarize_metrics(metrics_payload),
    }
    registry["versions"].append(entry)
    registry["latest_version"] = version
    save_registry(artifact_dir, registry)
    write_history_md(artifact_dir, registry)
    return version


def _fmt_money(v) -> str:
    return f"${v:,.0f}" if isinstance(v, (int, float)) else "-"


def _fmt_num(v, pct=False) -> str:
    if not isinstance(v, (int, float)):
        return "-"
    return f"{v:.1f}%" if pct else f"{v:.3f}"


def write_history_md(artifact_dir: Path, registry: Optional[Dict] = None) -> None:
    """(Re)generate the human-readable MODEL_HISTORY.md from the registry."""
    artifact_dir = Path(artifact_dir)
    registry = registry or load_registry(artifact_dir)
    lines = [
        "# Model history",
        "",
        "Every trained model is snapshotted into `archive/vN/` and listed here. The",
        "live model in `model_artifacts/` is always the latest version. Lower test",
        "MAE / MAPE and higher R² are better; interval coverage targets ~0.80.",
        "",
        "| Version | Date | Rows | Test MAE | R² | MAPE | q10–q90 coverage | Notes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for e in registry.get("versions", []):
        m = e.get("metrics", {})
        date = str(e.get("created_at", ""))[:10]
        cov = m.get("interval_coverage_q10_q90")
        cov_str = f"{cov:.2f}" if isinstance(cov, (int, float)) else "-"
        mape = m.get("test_mape_percent")
        lines.append(
            f"| v{e.get('version')} | {date} | {m.get('rows_total', '-')} | "
            f"{_fmt_money(m.get('test_mae'))} | {_fmt_num(m.get('test_r2'))} | "
            f"{_fmt_num(mape, pct=True)} | {cov_str} | {e.get('notes', '')} |"
        )
    lines.append("")
    (artifact_dir / HISTORY_NAME).write_text("\n".join(lines))
