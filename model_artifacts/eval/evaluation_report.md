# Model Evaluation Report

_Market-relative used-car pricing. Target = asking price from listings (not realized
sale price); all verdicts are relative to comparable listings on the market._

## 1. Model vs baselines (held-out test set)

| Method | MAE | R² | MAPE |
|---|---|---|---|
| **LightGBM (point)** | $3,045 | 0.795 | 21.8% |
| Global-median baseline | $10,658 | -0.319 | 50.5% |
| Same make/model median | $8,373 | 0.233 | 75.3% |
| Linear depreciation (age+miles+luxury) | $4,371 | 0.671 | 36.6% |

The model is compared with a simple baseline: the median price of the same make/model.

## 2. Where the model fails (segment analysis)

See `segment_errors.csv`. Error is not uniform: it concentrates in high-price /
luxury / older segments, where comparable listings are sparse.

```
  segment_type  segment   n     mae  mape_percent  band_coverage
    price_band     <15k 330  1132.0          23.7          0.767
    price_band   15-30k 112  3679.0          16.8          0.732
    price_band   30-50k  45  6413.0          17.5          0.733
    price_band   50-80k  13 10142.0          17.9          0.462
    price_band     80k+   7 48230.0          44.4          0.429
      age_band     0-2y  61  8624.0          18.1          0.689
      age_band     3-5y  65  5582.0          19.4          0.815
      age_band    6-10y  88  4144.0          25.3          0.591
      age_band     10y+ 293   990.0          22.0          0.785
mileage_bucket     high 242   930.0          22.5          0.760
mileage_bucket      low 106  6075.0          17.9          0.736
mileage_bucket   medium 116  2032.0          23.3          0.741
mileage_bucket very_low  43 10209.0          23.1          0.674
        luxury  economy 422  2270.0          21.2          0.749
        luxury   luxury  85  6893.0          24.4          0.718
```

## 3. Is the confidence band honest? (calibration)

[q10, q90] interval coverage on test: **74.4%** (target ~80%).
Quantile calibration (`calibration_curve.png`):

```
quantile  target_level  empirical_below
     q10           0.1            0.126
     q50           0.5            0.493
     q90           0.9            0.870
```

## 4. What drives price (SHAP)

Global importances (`shap_global.png`, `shap_global.csv`). Top features:

```
       feature  mean_abs_shap
          year       0.467448
       mileage       0.258081
           cpo       0.070908
       car_age       0.064190
    body_style       0.045913
    make_model       0.044061
          trim       0.040169
 engine_size_l       0.036701
    drivetrain       0.028914
is_luxury_make       0.027058
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
