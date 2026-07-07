# Quick Model Writeup

Generated: 2026-07-06 19:21:24

## Setup
- Model: lightgbm + quantile regression (q10/q50/q90) + mean point model
- Rows: total 12619, train 10095, test 2524
- Features: 21 (11 categorical)

## Headline results
- Test R2: 0.8260  (baseline R2: -0.0183)
- Test MAE: $3,068  (baseline MAE: $11,229)
- Test RMSE: $6,678
- Test MAPE: 12.83%

## Confidence band (q10–q90)
- Coverage on test set: 63.5% (ideal ~80%)
- Average band width: $7,238

## Top features (gain)
```
          feature  gain_importance  split_importance
       make_model     49116.498091             15606
          mileage     41997.000984             25436
   miles_per_year     36470.854349             27293
             trim     25502.938679             10737
             year     25408.221837              9955
    engine_size_l     14418.568992             10688
        car_color     10025.964815              7462
              mpg      7978.837877              6082
transmission_type      2950.159361              2308
       drivetrain      2831.511037              1721
```

## Sample predictions
- Closest call: predicted $23,878, actual $23,878, error $0
- Worst miss:  predicted $37,557, actual $132,489, error $94,932

### Listings that look like deals (actual < q10):
- listed $11,999, fair range $12,503–$16,355
- listed $15,998, fair range $16,222–$17,299
- listed $40,537, fair range $42,781–$56,582

### Listings that look overpriced (actual > q90):
- listed $19,900, fair range $16,086–$18,583
- listed $23,998, fair range $20,587–$22,867
- listed $39,998, fair range $26,650–$35,472
