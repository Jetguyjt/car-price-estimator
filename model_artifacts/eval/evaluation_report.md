# Model Evaluation Report

_Market-relative used-car pricing. Target = asking price from listings (not realized
sale price); all verdicts are relative to comparable listings on the market._

## 1. Model vs baselines (held-out test set)

| Method | MAE | R² | MAPE |
|---|---|---|---|
| **LightGBM (point)** | $8,059 | 0.422 | 22.4% |
| Global-median baseline | $12,504 | -0.073 | 42.2% |
| Same make/model median | $12,536 | -0.031 | 44.5% |
| Linear depreciation (age+miles+luxury) | $9,897 | 0.284 | 29.5% |

The model is compared with a simple baseline: the median price of the same make/model.

## 2. Where the model fails (segment analysis)

See `segment_errors.csv`. Error is not uniform: it concentrates in high-price /
luxury / older segments, where comparable listings are sparse.

```
  segment_type  segment   n     mae  mape_percent  band_coverage
    price_band     <15k  20  3112.0          27.5          0.450
    price_band   15-30k  86  4991.0          21.3          0.802
    price_band   30-50k  52  6208.0          15.9          0.750
    price_band   50-80k  22 18768.0          30.5          0.545
    price_band     80k+   4 63900.0          61.7          0.000
      age_band     0-2y  76  8005.0          20.0          0.789
      age_band     3-5y  54  9039.0          24.4          0.630
      age_band    6-10y  43  8283.0          24.3          0.651
      age_band     10y+  11  2740.0          22.3          0.636
mileage_bucket     high   8  3611.0          30.4          0.750
mileage_bucket      low  72 10218.0          24.4          0.694
mileage_bucket   medium  42  5461.0          22.4          0.524
mileage_bucket very_low  62  7884.0          19.2          0.823
        luxury  economy 143  6295.0          20.2          0.720
        luxury   luxury  41 14211.0          30.2          0.634
```

## 3. Is the confidence band honest? (calibration)

[q10, q90] interval coverage on test: **70.1%** (target ~80%).
Quantile calibration (`calibration_curve.png`):

```
quantile  target_level  empirical_below
     q10           0.1            0.152
     q50           0.5            0.495
     q90           0.9            0.853
```

## 4. What drives price (SHAP)

Global importances (`shap_global.png`, `shap_global.csv`). Top features:

```
       feature  mean_abs_shap
          year       0.178033
       mileage       0.130190
is_luxury_make       0.103708
    body_style       0.089185
miles_per_year       0.023046
          trim       0.021156
    make_model       0.015688
       car_age       0.015545
mileage_bucket       0.002598
        is_new       0.001095
```

The per-listing explainer (`make_listing_explainer`) returns signed contributions
for a single car — this is what the web UI surfaces as the human-readable 'why'.

## 5. External benchmark (KBB / Edmunds)

Created benchmark_kbb.csv with 30 rows to fill in by hand (look up each car's KBB/Edmunds value in the `kbb_value` column).

Comparing against an independent valuation source (KBB/Edmunds) is the strongest
available check that the model tracks real-world value, not just listing patterns.

## Known limitations

- **Asking, not sale price.** We never observe what a car actually sold for.
- **Data scale + features.** The collected sample is limited and uneven across makes,
  model lines, price ranges, and available detail fields.
- **Temporal snapshot.** Prices drift; the model reflects one collection window.
