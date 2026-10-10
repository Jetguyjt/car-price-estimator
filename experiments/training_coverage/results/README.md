# Saved aggregate results

This directory contains the compact tables and figures kept from the completed run. It does not contain listing records, fold assignments, row-level predictions, checkpoints, or fitted models.

## Main numbers

The random listing-level reference produced an MAE of $3,015 on 20,801 records. Removing complete mileage ranges from training raised MAE to $4,171, and removing complete vehicle-age ranges raised it to $4,403. Complete processed make/model and manufacturer holdouts produced MAEs of $7,676 and $7,921.

In the matched comparisons, the same test listings and number of training rows were used in both conditions. Removing the selected interior mileage and age ranges increased MAE by 19% and 15%. Removing the selected upper mileage and age tails increased MAE by 82% and 89%.

These values describe one processed sample and one fixed pipeline. They are not performance guarantees for other data.

## Figures

- [`evaluation_design_schematic.png`](figures/evaluation_design_schematic.png) shows the random, categorical, interior-range, and upper-tail conditions.
- [`primary_ratio_forest.png`](figures/primary_ratio_forest.png) compares pooled MAE ratios with the random listing-level reference.
- [`matched_range_comparison.png`](figures/matched_range_comparison.png) compares represented and absent training coverage for four matched ranges.

## Tables

The main tables are:

- [`dataset_summary.csv`](tables/dataset_summary.csv)
- [`evaluation_design.csv`](tables/evaluation_design.csv)
- [`primary_protocols.csv`](tables/primary_protocols.csv)
- [`matched_ranges.csv`](tables/matched_ranges.csv)
- [`sensitivity_checks.csv`](tables/sensitivity_checks.csv)

The other tables save checks for numerical ranges, category breadth, category weighting, matching quality, directional error, and repetition stability. [`frozen_run_manifest.json`](frozen_run_manifest.json) records the private dataset fingerprint, fixed model settings, software versions, and aggregate counts without exposing any listing rows or local paths. [`manifest.csv`](manifest.csv) lists each published result file with its SHA-256 digest.

## Metric notes

- **MAE** is the average absolute dollar error. Lower is better.
- **MAPE** is the average absolute error as a percentage of each asking price.
- **MAE ratio** divides one condition's MAE by its stated reference. A value above 1 means higher error.
- **Descriptive 95% interval** is the paired cluster-bootstrap interval for a difference or ratio. It is conditional on the saved fits and collected sample.

Ratios in `sensitivity_checks.csv` use each modified pipeline's own random reference. Ratios in `numerical_band_results.csv` use random-validation losses for the same numerical cohort. Those references are different from the equal-training-size represented controls in `matched_ranges.csv`.
