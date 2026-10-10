# Model history

Every trained model is listed here. Local training runs also save binary snapshots
under `archive/vN/`; those older binaries are not included in this checkout. The live
files in `model_artifacts/` are version 13. Lower evaluation MAE / MAPE and higher
R² are better; interval coverage targets about 0.80. Versions 10–13 reuse the same
fixed 20% evaluation split, which was reviewed during development. Those metrics are
descriptive development results, not estimates from a one-time untouched final test.

| Version | Date | Rows | Evaluation MAE | R² | MAPE | q10-q90 coverage | Notes |
|---|---|---|---|---|---|---|---|
| v1 | 2026-06-30 | 920 | $8,059 | 0.422 | 22.4% | 0.70 | baseline model (~900-row snapshot); imported as first tracked version |
| v2 | 2026-06-30 | 2535 | $3,045 | 0.795 | 21.8% | 0.74 | retrain on cleaned 2,535-row dataset (+8 detail-page features incl. engine_size_l, drivetrain, fuel, transmission); beats all baselines, R2 0.42->0.80 vs v1 (different test mix — see README caveat) |
| v3 | 2026-07-06 | 12619 | $3,068 | 0.826 | 12.8% | 0.64 | retrain on cleaned 12.6k-row dataset (5x v2). MAPE 21.8%->12.8% and R2 0.795->0.826 on a harder, 4.6x more expensive test set (median $23k vs $5k). Regression: q10-q90 coverage 74%->63% (band too narrow — recalibration is next). |
| v4 | 2026-07-08 | 12619 | $3,068 | 0.826 | 12.8% | 0.79 | conformalized quantile band (CQR): coverage 63%->79% (target 80%), same 12.6k data + identical point model as v3 so MAE/R2/MAPE unchanged. Fixes v3's band regression. See src/conformal.py + model_artifacts/conformal.json. |
| v5 | 2026-07-13 | 12739 | $3,132 | 0.840 | 13.0% | 0.81 | luxury-accuracy fix pt 1: stop deleting the price tail (absolute $500k cap replaces q99.5 cut; $/mile filter scoped to mileage>=2.5k and price<$80k), keep real trims (min count 10->5), add trim_tier feature (base/premium/performance). >$120k train rows 20->90; >$120k test MAPE 65%->28%. |
| v6 | 2026-07-13 | 12739 | $3,176 | 0.837 | 13.1% | 0.81 | repair scraper-truncated multi-word models (range rover*, grand cherokee, santa fe, amg <class>) so one make_model no longer spans a $5k Evoque and a $200k Range Rover. Shared with inference via repair_split_model(). |
| v7 | 2026-07-13 | 12739 | $2,947 | 0.854 | 12.6% | 0.80 | add smoothed make_model target encoding (mean log-price per model line, computed on proper-train slice, served from make_model_price_enc.json). Lets sparse expensive lines (AMG G, 911) escape shrinkage toward the market mean. Best metrics to date on a harder test set. |
| v8 | 2026-08-19 | 12739 | $2,947 | 0.854 | 12.6% | 0.80 | add predicted-price-bin (Mondrian) conformal adjustments, which reduced the measured coverage imbalance across price segments (conditional-coverage error 0.145->0.081, -44%). This is an empirical calibration improvement, not a guarantee for every segment. Point metrics are identical to v7. |
| v9 | 2026-08-19 | 12739 | $2,855 | 0.859 | 12.2% | 0.80 | add economic monotonicity constraints to the point model (price non-increasing in mileage and owner_count, non-decreasing in year). Eliminates tested ceteris-paribus violations at no accuracy cost; MAE $2,947->$2,855. Band quantiles stay unconstrained because LightGBM does not combine monotone constraints with quantile loss. |
| v10 | 2026-09-06 | 20801 | $3,023 | 0.881 | 11.0% | 0.79 | trained on 20801 rows |
| v11 | 2026-09-06 | 20801 | $2,978 | 0.885 | 11.0% | 0.79 | retrain on the expanded 20,801-listing dataset (~1.6x v9's 12.7k) and remove the sparse `region` tag from the feature set (present on only ~7% of rows, a leftover of an abandoned geographic probe; it carried no usable signal). Best metrics to date: MAE $2,978, R2 0.885, MAPE 11.0% on a larger, harder test set (n=4,161). Dropping region also edged MAE below v10's $3,023. Monotonicity constraints and group-conditional (Mondrian) conformal calibration carry over from v8/v9. |
| v12 | 2026-10-10 | 20801 | $2,973 | 0.885 | 11.0% | 0.79 | retrained on 20,801 rows with internal early-stopping validation; evaluation rows were untouched within this run, but the fixed split had been reviewed in earlier versions |
| v13 | 2026-10-10 | 20801 | $2,976 | 0.885 | 11.0% | 0.79 | retrained on 20,801 rows with fold-local target encoding and internal early-stopping validation; evaluation rows were untouched within this run, but the fixed split had been reviewed during development |
