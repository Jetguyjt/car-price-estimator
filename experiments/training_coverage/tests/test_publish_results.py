import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import coverage_shift.experiment as experiment_module
from coverage_shift.data import load_study_data, synthetic_listings
from coverage_shift.experiment import CoverageShiftExperiment, ExperimentConfig

try:
    from scripts.publish_results import (
        PUBLIC_SCOPES,
        build_frozen_run_manifest,
        publish_results,
    )
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from experiments.training_coverage.scripts.publish_results import (
        PUBLIC_SCOPES,
        build_frozen_run_manifest,
        publish_results,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.fixture(scope="module")
def completed_synthetic_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("publisher")
    run_dir = root / "completed-run"
    raw_path = root / "raw.csv"
    clean_path = root / "clean.csv"
    processed_path = root / "processed.csv"

    processed = synthetic_listings(n_rows=2_000, reference_year=2026)
    processed.to_csv(processed_path, index=False)
    processed.to_csv(clean_path, index=False)
    raw = processed.copy()
    raw["stock_type"] = np.where(raw["is_new"] == 1, "New", "Used")
    raw.to_csv(raw_path, index=False)

    data = load_study_data(processed_path, reference_year=2026)

    def quick_prediction(
        train_features,
        train_target,
        test_features,
        categorical_features,
        feature_order,
        **kwargs,
    ):
        return np.ones(len(test_features), dtype=float)

    original_fit_predict = experiment_module.fit_predict
    experiment_module.fit_predict = quick_prediction
    try:
        experiment = CoverageShiftExperiment(
            data,
            ExperimentConfig(
                output_dir=run_dir,
                reference_year=2026,
                bootstrap_reps=20,
                force=True,
            ),
        )
        experiment.run("all")
    finally:
        experiment_module.fit_predict = original_fit_predict

    return {
        "root": root,
        "run_dir": run_dir,
        "raw": raw_path,
        "clean": clean_path,
        "processed": processed_path,
    }


def _publish(paths: dict[str, Path], output_dir: Path) -> list[Path]:
    os.environ.setdefault("MPLCONFIGDIR", str(paths["root"] / "matplotlib"))
    return publish_results(
        run_dir=paths["run_dir"],
        raw_data=paths["raw"],
        clean_data=paths["clean"],
        processed_data=paths["processed"],
        output_dir=output_dir,
    )


def test_publish_results_builds_complete_verified_bundle(completed_synthetic_run):
    paths = completed_synthetic_run
    output_dir = paths["root"] / "published"
    generated = _publish(paths, output_dir)

    expected = {output_dir / path for path in PUBLIC_SCOPES}
    expected.add(output_dir / "manifest.csv")
    assert set(generated) == expected
    assert {
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_file()
    } == {*PUBLIC_SCOPES, "manifest.csv"}

    manifest = pd.read_csv(output_dir / "manifest.csv")
    assert manifest["path"].tolist() == sorted(PUBLIC_SCOPES)
    assert len(manifest) == 16
    for row in manifest.itertuples(index=False):
        assert _sha256(output_dir / row.path) == row.sha256

    frozen_path = output_dir / "frozen_run_manifest.json"
    frozen = json.loads(frozen_path.read_text())
    assert frozen["source_file"] == "private processed snapshot"
    assert frozen["source_sha256"] == _sha256(paths["processed"])
    assert frozen["n_rows"] == 2_000
    assert frozen["n_make_models"] == 32
    assert frozen["n_makes"] == 8

    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["study_sample"] == {
        "raw_records": 2_000,
        "records_after_primary_cleaning": 2_000,
        "model_ready_records": 2_000,
        "processed_make_model_categories": 32,
        "manufacturers": 8,
    }
    assert len(summary["primary_protocols"]) == 5
    assert len(summary["matched_ranges"]) == 4

    run_manifest = json.loads((paths["run_dir"] / "data_manifest.json").read_text())
    preserved, original_bytes = build_frozen_run_manifest(
        run_manifest=run_manifest,
        processed=pd.read_csv(paths["processed"]),
        processed_path=paths["processed"],
        supplied_manifest=frozen_path,
    )
    assert preserved == frozen
    assert original_bytes == frozen_path.read_bytes()


def test_publish_results_rejects_missing_run_file(completed_synthetic_run):
    paths = dict(completed_synthetic_run)
    broken_run = paths["root"] / "missing-file-run"
    shutil.copytree(paths["run_dir"], broken_run)
    (broken_run / "main_summary.csv").unlink()
    paths["run_dir"] = broken_run
    output_dir = paths["root"] / "missing-file-output"

    with pytest.raises(FileNotFoundError, match="main_summary.csv"):
        _publish(paths, output_dir)
    assert not output_dir.exists()


def test_publish_results_rejects_source_sha_mismatch(completed_synthetic_run):
    paths = dict(completed_synthetic_run)
    broken_run = paths["root"] / "sha-mismatch-run"
    shutil.copytree(paths["run_dir"], broken_run)
    manifest_path = broken_run / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    paths["run_dir"] = broken_run
    output_dir = paths["root"] / "sha-mismatch-output"

    with pytest.raises(ValueError, match="SHA-256"):
        _publish(paths, output_dir)
    assert not output_dir.exists()


def test_publish_results_rejects_source_count_mismatch(completed_synthetic_run):
    paths = dict(completed_synthetic_run)
    broken_run = paths["root"] / "count-mismatch-run"
    shutil.copytree(paths["run_dir"], broken_run)
    manifest_path = broken_run / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["n_rows"] += 1
    manifest_path.write_text(json.dumps(manifest))
    paths["run_dir"] = broken_run
    output_dir = paths["root"] / "count-mismatch-output"

    with pytest.raises(ValueError, match="row count"):
        _publish(paths, output_dir)
    assert not output_dir.exists()
