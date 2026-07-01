# Model history

Every trained model is snapshotted into `archive/vN/` and listed here. The
live model in `model_artifacts/` is always the latest version. Lower test
MAE / MAPE and higher R² are better; interval coverage targets ~0.80.

| Version | Date | Rows | Test MAE | R² | MAPE | q10–q90 coverage | Notes |
|---|---|---|---|---|---|---|---|
| v1 | 2026-06-30 | 920 | $8,059 | 0.422 | 22.4% | 0.70 | baseline model (~900-row snapshot); imported as first tracked version |
