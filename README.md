# CarPrice — market-relative used-car pricing

Use used-car listing data to train a LightGBM model on `log(price)` with **quantile
regression** for a price *band*, and judge any listing as **deal / fair / overpriced** —
with an honest evaluation of where the model works and where it doesn't.

> **What "deal / fair / overpriced" means.** The model learns from **asking prices** on
> listings, not realized sale prices. A verdict is therefore *market-relative*: priced low,
> in-line, or high **versus comparable listings** — not a claim about a car's true resale
> value. (Same basis as Zillow's Zestimate.)

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

This repository was initialized in **late June 2026** to preserve and publish the current working state of a local automotive ML research project. The project itself began earlier as local experiments and scripts around **late 2025**, before there was a formal GitHub history. As a result, the first commit is a repository baseline rather than the true beginning of the research work.

## Setup

```bash
python3 -m pip install -r requirements.txt
# macOS only: LightGBM needs the OpenMP runtime
brew install libomp
```


## Pipeline

```bash
# 1. Provide appropriately obtained raw listing data
#    -> writes cars_com_raw.csv

# 2. Clean + engineer features  (optional: pass input/output paths)
python3 preprocess_cars_data.py cars_com_raw_1000.csv cars_com_processed.csv
#    -> writes cars_com_processed.csv

# 3. Train LightGBM (quantile q10/q50/q90 + L2 point model)
python3 train_price_model.py
#    -> writes model_artifacts/

# 4. Rigorous evaluation (baselines, segments, calibration, SHAP, KBB scaffold)
python3 evaluate_model.py
#    -> writes model_artifacts/eval/

# 5. Score a single listing
python3 predict_listing.py     # edit EXAMPLE_LISTING inside the file
```

Run the tests:

```bash
python3 -m pytest tests/ -q
```

## What's in the repo

| Path | Purpose |
|---|---|
| `preprocess_cars_data.py` | Cleaning + feature engineering; normalizers are unit-tested. |
| `train_price_model.py` | LightGBM training; writes models + `metrics.json` + `category_levels.json`. |
| `evaluate_model.py` | Baselines, segment error analysis, calibration, SHAP, KBB benchmark scaffold. |
| `predict_listing.py` | Single-listing scoring: price band + verdict, with unseen-category warnings. |
| `tests/` | Unit tests for the preprocessing normalizers. |
| `model_artifacts/` | Trained models + metrics. `eval/` holds the evaluation report + plots. |
| `benchmark_kbb.csv` | Hand-fill template to benchmark the model against KBB/Edmunds. |

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

## Roadmap

This snapshot is the **ML + evaluation core**. Planned next:

1. **Scale the data** to 10–20k listings and populate detail-page fields (drivetrain, fuel,
   MPG, accidents) — the biggest lever on accuracy and the luxury blind spot.
2. attribute entry - map plain-language vehicle details to the
   model's schema (constrained by `category_levels.json`).
3. **Web app** — text box → extracted attributes → price band + verdict + SHAP "why".
4. **External benchmark** vs KBB/Edmunds; quantile recalibration for honest 80% coverage.

See `model_artifacts/eval/` for the full methodology and the limitations section.
