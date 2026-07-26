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

Current model is **v7**, trained on the cleaned 12,739-listing dataset. On a held-out
test set (n = 2,548), every method is scored on the **same** test rows:

| Method | MAE | R² | MAPE |
|---|---|---|---|
| **LightGBM (this model)** | **$2,947** | **0.854** | **12.6%** |
| Global-median baseline | $11,836 | −0.022 | 82.2% |
| Same make/model median | $8,761 | 0.497 | 60.8% |
| Linear depreciation baseline | $7,763 | 0.410 | 31.1% |

The model outperforms the three simple baselines evaluated here. The q10–q90 price band
is **conformally calibrated (CQR)** and covers **80%** of held-out prices, on target.
Error still concentrates in high-price / luxury cars, where comparable listings are sparse
(luxury-make MAPE 15.5% vs economy 11.8%), but far less than it used to: v5–v7 fixed a
preprocessing bug that silently deleted 80% of the >$120k cars from training, added a
trim-tier feature (base / premium / performance) so AMG/M/RS/Turbo variants stop pricing
like base models, and added a make_model target encoding so sparse expensive model lines
aren't shrunk toward the market average. Together those cut >$120k MAPE from **65% to
~30%** and the systematic underprediction bias at $60–80k from **−$12.9k to −$4.3k**.
More top-end listings remain the best next lever. Full breakdown:
`model_artifacts/eval/`.

> **On comparing versions.** MAE is *not* comparable across versions because each was
> tested on a differently-priced held-out set (v2's median $5.0k vs later versions' $23k).
> The honest, like-for-like story: the v2→v3 jump to 12.6k rows nearly halved relative
> error (**MAPE 21.8% → 12.8%**, R² 0.795 → 0.826) but left the band too narrow (63%
> coverage); **v4 added conformal calibration** for an honest ~80% band; **v5–v7 fixed
> luxury-segment accuracy** (restored the deleted price tail, trim-tier feature,
> split-model repair, make_model target encoding) and improved every headline metric on
> a strictly harder test set that now includes six-figure cars. Full changelog:
> [`model_artifacts/MODEL_HISTORY.md`](model_artifacts/MODEL_HISTORY.md).

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

### Local web app (attribute enterer)

A Streamlit front end wraps the model: pick a car's attributes and get the estimated
price, a conformally-calibrated fair-price band, and a deal / fair / overpriced verdict.
Dropdowns are populated from the model's own trained vocabularies, so an unseen make/model
or trim surfaces a low-confidence warning.

```bash
python3 -m streamlit run src/app.py     # opens http://localhost:8501
```

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
  predict_listing.py          single-listing scoring (+ assemble_listing helper)
  conformal.py                CQR band calibration
  app.py                      Streamlit attribute-enterer front end
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
| `src/` | Reusable Python code for cleaning, training, evaluation, and prediction. |
| `data/raw/` | Raw listing snapshots with source URLs and timestamps. |
| `data/interim/` | Cleaned or balanced intermediate datasets, including `cars_com_balanced_training_view.csv`. |
| `data/processed/` | Processed ML-ready datasets. |
| `data/snapshots/` | Smaller historical snapshots used during development. |
| `data/benchmarks/` | External benchmark templates such as KBB/Edmunds comparison scaffolds. |
| `notebooks/` | Exploratory notebooks for dataset profiling and model analysis. |
| `notes/` | Research notes and decision logs. |
| `media/` | Portfolio images, diagrams, screenshots, and demo assets. |
| `outputs/` | Exported portfolio/report outputs that should not break the training pipeline. |
| `dev/probes/` | One-off browser/CDP probes used while debugging Cars.com rendering. |
| `tests/` | Unit tests for the preprocessing normalizers. |
| `docs/dataset_profile_2026-07-01.md` | Point-in-time July 1 raw dataset profile and balancing diagnosis. |
| `docs/portfolio_summary.md` | Point-in-time portfolio summary from the initial repository setup. |
| `model_artifacts/` | Trained models + metrics. `eval/` holds the evaluation report + plots. |

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
  saved to `model_artifacts/category_levels.json`, and their per-value counts to
  `category_counts.json`, so a make/model that is unseen **or too sparse** (e.g. a BMW M3
  with only ~5 listings) is flagged as **low-confidence** rather than silently guessed.
- **Leakage:** `price_per_mile` is engineered for analysis but dropped before training.
- **Versioning:** every training run snapshots its artifacts into `model_artifacts/archive/vN/`
  and appends a row to [`model_artifacts/MODEL_HISTORY.md`](model_artifacts/MODEL_HISTORY.md),
  so improving the model never overwrites the one it replaced. The live model in
  `model_artifacts/` is always the latest version.

## Roadmap

Done so far:

- ✅ **Scaled the data** to ~12.6k cleaned listings with detail-page fields.
- ✅ **Honest confidence band** — conformal (CQR) calibration to ~80% coverage.
- ✅ **Local web app** — Streamlit attribute-enterer → price band + verdict + warnings.

Planned next:

1. **Close the luxury blind spot** — targeted collection of high-price listings (the
   $80k+ segment is still the weakest, ~$32k MAE on few comparables).
2. attribute entry - map plain-language vehicle details to the
   model's schema (constrained by `category_levels.json`) and feeds the same web app.
3. **Per-listing "why"** — surface SHAP contributions in the app, not just warnings.
4. **External benchmark** vs KBB/Edmunds on a hand-collected sample.

See `model_artifacts/eval/` for the full methodology and the limitations section.
