# Quick Model Writeup

Generated: 2026-07-13 21:09:30

## Setup
- Model: lightgbm + quantile regression (q10/q50/q90) + mean point model
- Rows: total 12739, train 10191, test 2548
- Features: 22 (11 categorical)

## Headline results
- Test R2: 0.8401  (baseline R2: -0.0221)
- Test MAE: $3,132  (baseline MAE: $11,836)
- Test RMSE: $7,828
- Test MAPE: 12.99%

## Confidence band (q10–q90)
- Coverage on test set: 80.7% (ideal ~80%)
- Average band width: $10,900

## Top features (gain)
```
          feature  gain_importance  split_importance
       make_model     40670.833725             11331
          mileage     32814.940963             17844
   miles_per_year     25752.478686             18731
             year     22850.642730              7326
             trim     19884.046903              7776
    engine_size_l     10758.068472              7570
        car_color      7353.326092              5310
              mpg      6031.218235              4423
       drivetrain      2332.215137              1348
transmission_type      2131.065162              1635
```

## Sample predictions
- Closest call: predicted $25,999, actual $25,998, error $1
- Worst miss:  predicted $76,998, actual $239,991, error $162,993

### Listings that look like deals (actual < q10):
- listed $4,357, fair range $4,657–$6,796
- listed $3,495, fair range $3,941–$10,502
- listed $18,925, fair range $19,011–$28,618

### Listings that look overpriced (actual > q90):
- listed $33,500, fair range $2,858–$11,644
- listed $45,998, fair range $23,677–$37,855
- listed $37,407, fair range $24,916–$37,050
