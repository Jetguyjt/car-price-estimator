import json

import pytest

pytest.importorskip("lightgbm")

from coverage_shift.data import prepare_study_data, synthetic_listings
from coverage_shift.experiment import CoverageShiftExperiment, ExperimentConfig


def test_synthetic_smoke_run_writes_finite_metrics(tmp_path):
    frame = synthetic_listings(n_rows=120, reference_year=2026)
    data = prepare_study_data(frame, reference_year=2026)
    experiment = CoverageShiftExperiment(
        data,
        ExperimentConfig(output_dir=tmp_path, reference_year=2026, bootstrap_reps=20),
    )

    experiment.run("smoke")

    result = json.loads((tmp_path / "smoke_test.json").read_text())
    manifest = json.loads((tmp_path / "data_manifest.json").read_text())
    assert result["n"] == 120
    assert result["mae"] > 0
    assert len(result["folds"]) == 2
    assert manifest["reference_year"] == 2026
    assert manifest["source_file"] == "synthetic"
