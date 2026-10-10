# Training-coverage experiments

This folder tests how one vehicle asking-price model behaves when parts of the input space are missing from training. It includes the experiment code, aggregate output tables, figures, and tests that run on generated data.

The completed run used 20,801 vehicle listings. Prices are advertised asking prices, not completed-sale prices.

## What was tested

Five evaluation conditions used the same LightGBM feature and model pipeline:

1. Random listing-level folds, where most categories and numerical ranges remained represented in training.
2. Complete mileage ranges removed from training.
3. Complete vehicle-age ranges removed from training.
4. Complete processed make/model categories removed from training.
5. Complete manufacturers removed from training.

Every eligible listing received an out-of-fold prediction. The random, make/model, and manufacturer conditions were repeated with three fold-assignment seeds. A second comparison kept both the test listings and training-row count fixed for selected mileage and age ranges.

The full setup is in [`docs/experiment_design.md`](docs/experiment_design.md).

## Saved results

| Evaluation condition | MAE | Relative to random | MAPE |
|---|---:|---:|---:|
| Random listing-level reference | $3,015 | 1.00× | 10.9% |
| Complete mileage ranges withheld | $4,171 | 1.38× | 18.4% |
| Complete vehicle-age ranges withheld | $4,403 | 1.46× | 23.0% |
| Complete make/model categories withheld | $7,676 | 2.55× | 25.4% |
| Complete manufacturers withheld | $7,921 | 2.63× | 25.7% |

![MAE ratios across evaluation conditions](results/figures/primary_ratio_forest.png)

The matched comparisons gave a more controlled look at numerical gaps. With the same test rows and training-row count in both conditions, removing the selected interior mileage and age ranges raised MAE by about 19% and 15%. Removing the high-mileage and older-vehicle tails raised MAE by about 82% and 89%.

These numbers describe one processed sample and one model pipeline. They are not universal error rates.

## Folder contents

```text
data/                 input schema and data notes
docs/                 experiment design
results/tables/       saved aggregate tables
results/figures/      figures built from those tables
src/coverage_shift/   preparation, splitting, modeling, and metrics
scripts/              result validation and figure generation
tests/                tests that use generated data only
```

No row-level listings, row-level predictions, checkpoints, or fitted model files are included here.

## Setup and checks

The saved run used Python 3.13.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
make test
make smoke
python scripts/validate_results.py
```

`make smoke` creates a deterministic synthetic dataset and runs a short end-to-end check. Its output is only a software test, not one of the saved results.

Rebuild the committed figures from the aggregate tables with:

```bash
python scripts/make_figures.py
```

Run the full experiment on a processed table that follows [`data/schema.csv`](data/schema.csv):

```bash
PYTHONPATH=src python -m coverage_shift \
  --data /path/to/vehicle_listings_processed.csv \
  --output-dir outputs/reproduced \
  --reference-year 2026 \
  --stage all
```

The full run fits 184 models and can take more than an hour on a laptop. It can also be run as `main`, `paired`, and `sensitivity` stages with the same output directory. Completed checkpoints are reused unless `--force` is supplied.

After all three stages finish, regenerate the aggregate result bundle with:

```bash
make publish-results \
  RUN_DIR=outputs/reproduced \
  RAW_DATA=/path/to/vehicle_listings_raw.csv \
  CLEAN_DATA=/path/to/vehicle_listings_clean.csv \
  PROCESSED_DATA=/path/to/vehicle_listings_processed.csv
```

The default destination is `build/published-results`, so this command does not replace the committed results. The publisher checks the processed-data fingerprint and aggregate counts against the completed run, requires every full-run output, rebuilds all 11 tables, the summary, three figures, the sanitized frozen-run metadata, and the checksum manifest, and stops without promoting partial output if validation fails.

A run created by the current code contains everything needed to regenerate `frozen_run_manifest.json`. For an older completed run whose `data_manifest.json` predates the current provenance fields, provide a matching sanitized manifest explicitly:

```bash
make publish-results \
  RUN_DIR=/path/to/completed-run \
  RAW_DATA=/path/to/vehicle_listings_raw.csv \
  CLEAN_DATA=/path/to/vehicle_listings_clean.csv \
  PROCESSED_DATA=/path/to/vehicle_listings_processed.csv \
  FROZEN_RUN_MANIFEST=/path/to/frozen_run_manifest.json
```

The supplied metadata is accepted only when its dataset fingerprint, row and group counts, and all overlapping run fields agree with the completed run.

## Data boundary

The original listing-level data are not redistributed because they came from a third-party marketplace. Exact numerical reproduction requires the same frozen sample. Running this code on another appropriately obtained dataset is a new comparison and should be expected to produce different values.

The saved intervals describe variation within the collected sample and fitted models. They do not include the uncertainty that would come from collecting a different sample, retraining in a different time period, or using another marketplace.
