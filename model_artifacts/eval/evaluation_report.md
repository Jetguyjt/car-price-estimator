# Model Evaluation Report

_Market-relative used-car pricing. Target = asking price from listings (not realized
sale price); all verdicts are relative to comparable listings on the market._

## 1. Model vs baselines (held-out test set)

| Method | MAE | R² | MAPE |
|---|---|---|---|
| **LightGBM (point)** | $3,068 | 0.826 | 12.8% |
| Global-median baseline | $11,229 | -0.018 | 83.8% |
| Same make/model median | $8,499 | 0.391 | 59.9% |
| Linear depreciation (age+miles+luxury) | $6,847 | 0.508 | 29.3% |

The model is compared with a simple baseline: the median price of the same make/model.

## 2. Where the model fails (segment analysis)

See `segment_errors.csv`. Error is not uniform: it concentrates in high-price /
luxury / older segments, where comparable listings are sparse.

```
  segment_type  segment    n     mae  mape_percent  band_coverage
    price_band     <15k  648  1594.0          20.7          0.776
    price_band   15-30k 1184  2025.0           9.1          0.820
    price_band   30-50k  538  3620.0           9.5          0.846
    price_band   50-80k  123 11118.0          17.9          0.553
    price_band     80k+   31 32183.0          30.9          0.258
      age_band     0-2y  539  4007.0           9.1          0.814
      age_band     3-5y  633  3680.0          10.1          0.798
      age_band    6-10y  937  2577.0          13.0          0.796
      age_band     10y+  415  2023.0          21.4          0.759
mileage_bucket     high  419  2092.0          20.6          0.745
mileage_bucket      low  977  3541.0          10.1          0.820
mileage_bucket   medium  811  2167.0          12.9          0.795
mileage_bucket very_low  317  5205.0          10.6          0.779
        luxury  economy 1977  2238.0          11.7          0.795
        luxury   luxury  547  6069.0          16.7          0.793
```

## 3. Is the confidence band honest? (calibration)

[q10, q90] interval coverage on test: **79.4%** (target ~80%).
Quantile calibration (`calibration_curve.png`):

```
quantile  target_level  empirical_below
     q10           0.1            0.091
     q50           0.5            0.495
     q90           0.9            0.885
```

## 4. What drives price (SHAP)

Global importances (`shap_global.png`, `shap_global.csv`). Top features:

```
       feature  mean_abs_shap
          year       0.277286
       mileage       0.197719
    make_model       0.154853
          trim       0.049831
is_luxury_make       0.045559
 engine_size_l       0.033132
    body_style       0.023301
    drivetrain       0.023220
       car_age       0.020187
miles_per_year       0.008864
```

The per-listing explainer (`make_listing_explainer`) returns signed contributions
for a single car — this is what the web UI surfaces as the human-readable 'why'.

## 5. External benchmark (KBB / Edmunds)

./data/benchmarks/benchmark_kbb.csv exists but no kbb_value rows are filled in yet.

Comparing against an independent valuation source (KBB/Edmunds) is the strongest
available check that the model tracks real-world value, not just listing patterns.

## Known limitations

- **Asking, not sale price.** We never observe what a car actually sold for.
- **Data scale + features.** The collected sample is limited and uneven across makes,
  model lines, price ranges, and available detail fields.
- **Temporal snapshot.** Prices drift; the model reflects one collection window.
