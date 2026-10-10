# Model Evaluation Report

_Market-relative vehicle pricing. Target = asking price from listings (not realized
sale price); all verdicts are relative to comparable listings on the market._

## 1. Model vs baselines (fixed evaluation split)

| Method | MAE | R² | MAPE |
|---|---|---|---|
| **LightGBM (point)** | $2,976 | 0.885 | 11.0% |
| Global-median baseline | $12,753 | -0.032 | 67.0% |
| Same make/model median | $9,049 | 0.573 | 49.7% |
| Linear depreciation (age+miles+luxury) | $8,645 | 0.329 | 29.4% |

The relevant question is not only whether R² is high, but whether the model beats
a simple comparable-vehicle lookup: the median price of the same make/model.

## 2. Where the model fails (segment analysis)

See `segment_errors.csv`. Error is not uniform: it concentrates in high-price /
luxury / older segments, where comparable listings are sparse.

```
  segment_type  segment    n     mae  mape_percent  band_coverage
    price_band     <15k  740  1777.0          23.9          0.761
    price_band   15-30k 1942  1718.0           7.7          0.818
    price_band   30-50k 1129  2715.0           7.1          0.800
    price_band   50-80k  259  7811.0          12.8          0.718
    price_band     80k+   91 29060.0          21.0          0.582
      age_band     0-2y 1341  3775.0           7.4          0.810
      age_band     3-5y 1078  2600.0           8.5          0.798
      age_band    6-10y 1209  2575.0          11.4          0.774
      age_band     10y+  533  2640.0          24.4          0.771
mileage_bucket     high  450  2003.0          21.0          0.762
mileage_bucket      low 1771  2664.0           8.2          0.807
mileage_bucket   medium 1193  2207.0          12.8          0.780
mileage_bucket very_low  747  5530.0           9.0          0.791
        luxury  economy 3282  2191.0          10.3          0.805
        luxury   luxury  879  5908.0          13.6          0.739
```

## 3. Is the confidence band honest? (calibration)

[q10, q90] interval coverage on the evaluation split: **79.1%** (target ~80%).
Raw quantile diagnostics (`calibration_curve.png`):

```
quantile  target_level  empirical_below
     q10           0.1            0.191
     q50           0.5            0.501
     q90           0.9            0.816
```

## 4. What drives price (SHAP)

Global importances (`shap_global.png`, `shap_global.csv`). Top features:

```
             feature  mean_abs_shap
                year       0.240381
             mileage       0.216432
make_model_price_enc       0.195676
          make_model       0.078016
                trim       0.047871
             car_age       0.033603
       engine_size_l       0.025026
      miles_per_year       0.024609
           trim_tier       0.021162
      is_luxury_make       0.020932
```

The reusable `make_listing_explainer` function returns signed contributions
for one listing in log-price space.

## Known limitations

- **Asking, not sale price.** We never observe what a car actually sold for.
- **Data scale + coverage.** The six-figure/luxury segment is thinly represented,
  and this is where error concentrates; collecting more high-price and rare-model
  listings is the top driver of future accuracy.
- **Temporal snapshot.** Prices drift; the model reflects one collection window.
- **Dataset-wide preprocessing.** Price-based outlier rules and rare-category
  grouping based on full-sample counts were applied before the 80/20 split.
- **Repeated evaluation use.** The fixed split was reviewed across versions 10-13
  and influenced feature selection during development. Its rows were not used to
  fit or calibrate version 13, but this is not a one-time independent final test.
