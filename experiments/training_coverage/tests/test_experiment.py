import numpy as np
import pandas as pd

from coverage_shift.data import prepare_study_data, synthetic_listings
from coverage_shift.experiment import CoverageShiftExperiment, ExperimentConfig


def test_checkpoint_identity_includes_reference_year(tmp_path):
    frame = synthetic_listings(n_rows=60, reference_year=2026)
    data_2026 = prepare_study_data(frame, reference_year=2026)
    data_2025 = prepare_study_data(frame, reference_year=2025)

    experiment_2026 = CoverageShiftExperiment(
        data_2026, ExperimentConfig(output_dir=tmp_path, reference_year=2026)
    )
    experiment_2025 = CoverageShiftExperiment(
        data_2025, ExperimentConfig(output_dir=tmp_path, reference_year=2025)
    )

    assert experiment_2026.data_tag != experiment_2025.data_tag


def test_all_registered_protocols_write_complete_outputs(tmp_path, monkeypatch):
    frame = synthetic_listings(n_rows=250, reference_year=2026)
    data = prepare_study_data(frame, reference_year=2026)

    def quick_prediction(
        train_features,
        train_target,
        test_features,
        categorical_features,
        feature_order,
        **kwargs,
    ):
        return np.ones(len(test_features), dtype=float)

    monkeypatch.setattr("coverage_shift.experiment.fit_predict", quick_prediction)
    experiment = CoverageShiftExperiment(
        data,
        ExperimentConfig(output_dir=tmp_path, bootstrap_reps=20, force=True),
    )

    experiment.run("all")

    main = pd.read_csv(tmp_path / "main_summary.csv")
    folds = pd.read_csv(tmp_path / "main_fold_metrics.csv")
    paired = pd.read_csv(tmp_path / "paired_range_summary.csv")
    sensitivity = pd.read_csv(tmp_path / "sensitivity_summary.csv")

    assert set(main["protocol"]) == {
        "random_reference",
        "holdout_make_model",
        "holdout_make",
        "holdout_mileage_primary",
        "holdout_age_primary",
        "holdout_mileage_sensitivity",
        "holdout_age_sensitivity",
    }
    assert len(folds) == 65
    assert len(paired) == 4
    assert set(paired["region_type"]) == {"interior", "upper_tail"}
    assert set(sensitivity["analysis_set"]) == {
        "no_price_encoding",
        "strict_exact_content_dedup",
        "mileage_proxy_reduced",
        "manufacturer_aware_model",
    }
    assert (paired["mae_ratio"] == 1).all()
