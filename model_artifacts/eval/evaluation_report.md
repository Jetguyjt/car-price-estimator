# Model Evaluation Report

_Market-relative used-car pricing. Target = asking price from listings (not realized
sale price); all verdicts are relative to comparable listings on the market._

## 1. Model vs baselines (held-out test set)

| Method | MAE | R² | MAPE |
|---|---|---|---|
| **LightGBM (point)** | $2,947 | 0.854 | 12.6% |
| Global-median baseline | $11,836 | -0.022 | 82.2% |
| Same make/model median | $8,761 | 0.497 | 60.8% |
| Linear depreciation (age+miles+luxury) | $7,763 | 0.410 | 31.1% |

The model is compared with a simple baseline: the median price of the same make/model.

## 2. Where the model fails (segment analysis)

See `segment_errors.csv`. Error is not uniform: it concentrates in high-price /
luxury / older segments, where comparable listings are sparse.

```
  segment_type  segment    n     mae  mape_percent  band_coverage
    price_band     <15k  600  1664.0          24.5          0.718
    price_band   15-30k 1224  1778.0           8.0          0.876
    price_band   30-50k  562  3364.0           8.7          0.847
    price_band   50-80k  123  7949.0          12.9          0.553
    price_band     80k+   39 37564.0          28.1          0.308
      age_band     0-2y  562  3905.0           9.0          0.829
      age_band     3-5y  650  3235.0           8.3          0.838
      age_band    6-10y  938  2606.0          12.0          0.826
      age_band     10y+  398  1926.0          26.3          0.686
mileage_bucket     high  371  1706.0          21.8          0.717
mileage_bucket      low  984  3182.0           9.5          0.840
mileage_bucket   medium  876  2171.0          13.1          0.809
mileage_bucket very_low  317  5810.0          10.3          0.811
        luxury  economy 2008  2127.0          11.8          0.820
        luxury   luxury  540  5994.0          15.5          0.763
```

## 3. Is the confidence band honest? (calibration)

[q10, q90] interval coverage on test: **80.8%** (target ~80%).
Quantile calibration (`calibration_curve.png`):

```
quantile  target_level  empirical_below
     q10           0.1            0.096
     q50           0.5            0.512
     q90           0.9            0.904
```

## 4. What drives price (SHAP)

Global importances (`shap_global.png`, `shap_global.csv`). Top features:

```
             feature  mean_abs_shap
                year       0.275388
             mileage       0.199140
make_model_price_enc       0.176994
          make_model       0.072293
                trim       0.043511
       engine_size_l       0.035623
      is_luxury_make       0.028240
           trim_tier       0.022502
          drivetrain       0.015741
          body_style       0.011292
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
