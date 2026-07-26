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
| v5 | 2026-07-13 | 12739 | $3,132 | 0.840 | 13.0% | 0.81 | luxury-accuracy fix pt 1: stop deleting the price tail (absolute $500k cap replaces q99.5 cut; $/mile filter scoped to mileage>=2.5k and price<$80k), keep real trims (min count 10->5), add trim_tier feature (base/premium/performance). >$120k train rows 20->90; >$120k test MAPE 65%->28%. |
| v6 | 2026-07-13 | 12739 | $3,176 | 0.837 | 13.1% | 0.81 | repair scraper-truncated multi-word models (range rover*, grand cherokee, santa fe, amg <class>) so one make_model no longer spans a $5k Evoque and a $200k Range Rover. Shared with inference via repair_split_model(). |
| v7 | 2026-07-13 | 12739 | $2,947 | 0.854 | 12.6% | 0.80 | add smoothed make_model target encoding (mean log-price per model line, computed on proper-train slice, served from make_model_price_enc.json). Lets sparse expensive lines (AMG G, 911) escape shrinkage toward the market mean. Best metrics to date on a harder test set. |
