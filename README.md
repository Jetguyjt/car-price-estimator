# CarPrice — market-relative used-car pricing

Use used-car listing data to train a LightGBM model on `log(price)` with **quantile
regression** for a price *band*, and judge any listing as **deal / fair / overpriced** —
with an honest evaluation of where the model works and where it doesn't.

> **What "deal / fair / overpriced" means.** The model learns from **asking prices** on
> listings, not realized sale prices. A verdict is therefore *market-relative*: priced low,
> in-line, or high **versus comparable listings** — not a claim about a car's true resale
> value. (Same basis as Zillow's Zestimate.)

## Portfolio highlights

- **End-to-end ML pipeline:** listing-data intake → raw cleaning → feature engineering → LightGBM training → evaluation → single-listing scoring.
- **Defensive data cleaning:** a dedicated `raw → interim` stage catches scrape artifacts a naive price floor misses — e.g. captured *monthly-payment* amounts surfacing as a $1,892 "2027 Corvette" — using a market-plausibility rule, and logs every drop to an auditable report.
- **Traceable dataset design:** raw listing URLs, listing keys, scrape timestamps, and detail-page enrichment timestamps are preserved in the raw CSV.
- **Uncertainty-aware predictions:** quantile models estimate a q10/q50/q90 price band, so the system gives a range as well as a point estimate.
- **Explainability:** model artifacts include feature importance, segment error analysis, calibration outputs, and SHAP summaries.
- **Honest limitations:** the project models asking prices, documents weak segments, and avoids filling missing data with guesses.

## Results at a glance

On a held-out test set (n = 184, ~900-listing snapshot):

| Method | MAE | R² | MAPE |
|---|---|---|---|
| **LightGBM (this model)** | **$8,059** | **0.422** | **22.4%** |
| Global-median baseline | $12,504 | −0.073 | 42.2% |
| Same make/model median | $12,536 | −0.031 | 44.5% |
| Linear depreciation baseline | $9,897 | 0.284 | 29.5% |

The model beats every baseline, but error is **not uniform**: it concentrates in luxury /
high-price cars (luxury MAE $14.2k vs economy $6.3k), where comparable listings are sparse.
The q10–q90 band currently covers 70% of test prices (target ~80% — slightly too narrow).
Full breakdown: `model_artifacts/eval/`.

## Project timeline

This repository was initialized in **late June 2026** to preserve and publish the current working state of a local automotive ML research project. The project itself began earlier as local experiments and scripts around **November 2025**, before there was a formal GitHub history. As a result, the first commit is a repository baseline rather than the true beginning of the research work, and the earlier timeline is reconstructed from the local files rather than a commit log.


## Setup

```bash
python3 -m pip install -r requirements.txt
# macOS only: LightGBM needs the OpenMP runtime
brew install libomp
```


## Pipeline

Paths are resolved from the project root via `src/paths.py`. Run the commands
below from the repository root.

```bash
# 1. Provide appropriately obtained raw listing data
#    -> writes data/raw/cars_com_raw.csv

# 2. Clean the raw scrape (dedup, drop impossible values, kill payment-amount
#    artifacts). Writes a per-reason report you can audit.
python3 src/clean_raw_data.py
#    -> writes data/interim/cars_com_clean.csv + data/interim/cleaning_report.md

# 3. Engineer features (normalization, feature columns, rare-category grouping)
python3 src/preprocess_cars_data.py
#    -> writes data/processed/cars_com_processed.csv

# 4. Train LightGBM (quantile q10/q50/q90 + L2 point model)
python3 src/train_price_model.py
#    -> writes model_artifacts/

# 5. Rigorous evaluation (baselines, segments, calibration, SHAP, KBB scaffold)
python3 src/evaluate_model.py
#    -> writes model_artifacts/eval/

# 6. Score a single listing
python3 src/predict_listing.py     # edit EXAMPLE_LISTING inside the file
```

Steps 2 and 3 accept optional `input.csv output.csv` overrides; with no arguments
they use the `data/` layout above.

Run the tests:

```bash
python3 -m pytest tests/ -q
```

## Repository layout

```
src/                          pipeline code (paths resolved via src/paths.py)
  clean_raw_data.py           raw hygiene: dedup, impossible values, artifacts
  preprocess_cars_data.py     normalization + feature engineering
  train_price_model.py        LightGBM quantile + point training
  evaluate_model.py           baselines, segments, calibration, SHAP
  predict_listing.py          single-listing scoring
  paths.py                    single source of truth for all data/artifact paths
data/
  raw/         immutable scraper output (cars_com_raw.csv)
  interim/     cleaned raw + cleaning_report.md
  processed/   model-ready table (cars_com_processed.csv)
  snapshots/   frozen point-in-time CSVs
  benchmarks/  KBB/Edmunds benchmark template
model_artifacts/  trained models + metrics; eval/ holds report + plots
docs/   tests/   dev/probes/ (throwaway debug scripts)
```

| Path | Purpose |
|---|---|
| `src/clean_raw_data.py` | Raw hygiene: dedup, impossible-value + payment-artifact removal; writes a cleaning report. |
| `src/preprocess_cars_data.py` | Normalization + feature engineering; normalizers are unit-tested. |
| `src/train_price_model.py` | LightGBM training; writes models + `metrics.json` + `category_levels.json`. |
| `src/evaluate_model.py` | Baselines, segment error analysis, calibration, SHAP, KBB benchmark scaffold. |
| `src/predict_listing.py` | Single-listing scoring: price band + verdict, with unseen-category warnings. |
| `tests/` | Unit tests for the preprocessing normalizers. |
| `docs/portfolio_summary.md` | Point-in-time portfolio summary from the initial repository setup. |
| `model_artifacts/` | Trained models + metrics. `eval/` holds the evaluation report + plots. |
| `data/benchmarks/benchmark_kbb.csv` | Hand-fill template to benchmark the model against KBB/Edmunds. |

## Data collection principles

The raw data collection policy is intentionally conservative:

- only save values visible on loaded Cars.com pages;
- leave unknown fields blank instead of inferring them;
- deduplicate by Cars.com `listing_key`;
- preserve timestamps and source URLs;
- pause between page loads and stop rather than brute-force blocks or CAPTCHA.


## Model design notes

- **Target:** trained on `log1p(price)`; predictions inverse-transformed.
- **Quantile band:** three models (q10/q50/q90) give a range; a listing above q90 reads as
  overpriced, below q10 as a deal. The quantiles are **sorted per-prediction** so the band
  never crosses.
- **Point estimate:** a separate L2 model (better MAE/R² than q50 alone).
- **Categoricals:** LightGBM handles them natively. The exact levels seen in training are
  saved to `model_artifacts/category_levels.json` so an unseen make/model or trim at
  prediction time is flagged as **low-confidence** rather than silently guessed.
- **Leakage:** `price_per_mile` is engineered for analysis but dropped before training.
- **Versioning:** every training run snapshots its artifacts into `model_artifacts/archive/vN/`
  and appends a row to [`model_artifacts/MODEL_HISTORY.md`](model_artifacts/MODEL_HISTORY.md),
  so improving the model never overwrites the one it replaced. The live model in
  `model_artifacts/` is always the latest version.

## Roadmap

This snapshot is the **ML + evaluation core**. Planned next:

1. **Scale the data** to 10–20k listings and populate detail-page fields (drivetrain, fuel,
   MPG, accidents) — the biggest lever on accuracy and the luxury blind spot.
2. attribute entry - map plain-language vehicle details to the
   model's schema (constrained by `category_levels.json`).
3. **Web app** — text box → extracted attributes → price band + verdict + SHAP "why".
4. **External benchmark** vs KBB/Edmunds; quantile recalibration for honest 80% coverage.

See `model_artifacts/eval/` for the full methodology and the limitations section.
