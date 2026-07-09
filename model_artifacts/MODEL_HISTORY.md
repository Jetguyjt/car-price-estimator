# Model history

Every trained model is snapshotted into `archive/vN/` and listed here. The
live model in `model_artifacts/` is always the latest version. Lower test
MAE / MAPE and higher R² are better; interval coverage targets ~0.80.

| Version | Date | Rows | Test MAE | R² | MAPE | q10–q90 coverage | Notes |
|---|---|---|---|---|---|---|---|
| v1 | 2026-06-30 | 920 | $8,059 | 0.422 | 22.4% | 0.70 | baseline model (~900-row snapshot); imported as first tracked version |
| v2 | 2026-06-30 | 2535 | $3,045 | 0.795 | 21.8% | 0.74 | retrain on cleaned 2,535-row dataset (+8 detail-page features incl. engine_size_l, drivetrain, fuel, transmission); beats all baselines, R2 0.42->0.80 vs v1 (different test mix — see README caveat) |
| v3 | 2026-07-06 | 12619 | $3,068 | 0.826 | 12.8% | 0.64 | retrain on cleaned 12.6k-row dataset (5x v2). MAPE 21.8%->12.8% and R2 0.795->0.826 on a harder, 4.6x more expensive test set (median $23k vs $5k). Regression: q10-q90 coverage 74%->63% (band too narrow — recalibration is next). |
| v4 | 2026-07-08 | 12619 | $3,068 | 0.826 | 12.8% | 0.79 | conformalized quantile band (CQR): coverage 63%->79% (target 80%), same 12.6k data + identical point model as v3 so MAE/R2/MAPE unchanged. Fixes v3's band regression. See src/conformal.py + model_artifacts/conformal.json. |
