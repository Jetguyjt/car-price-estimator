# Quick Model Writeup

Generated: 2026-06-30 20:10:57

## Setup
- Model: lightgbm + quantile regression (q10/q50/q90) + mean point model
- Rows: total 2535, train 2028, test 507
- Features: 21 (11 categorical)

## Headline results
- Test R2: 0.7954  (baseline R2: -0.3191)
- Test MAE: $3,045  (baseline MAE: $10,658)
- Test RMSE: $7,775
- Test MAPE: 21.78%

## Confidence band (q10-q90)
- Coverage on test set: 74.2% (ideal ~80%)
- Average band width: $8,029

## Top features (gain)
```
          feature  gain_importance  split_importance
          mileage      8058.768274              5995
             year      7687.702940              3023
   miles_per_year      4370.199034              5960
    engine_size_l      2348.680328              2750
       make_model      2141.114743               725
             trim      1188.143834               772
          car_age      1034.077700               315
        car_color       709.149108               629
              cpo       539.965766               360
transmission_type       506.373146               681
```

## Sample predictions
- Closest call: predicted $3,898, actual $3,900, error $2
- Worst miss:  predicted $20,208, actual $119,800, error $99,592

### Listings that look like deals (actual < q10):
- listed $1,808, fair range $3,535-$6,178
- listed $3,495, fair range $4,007-$5,556
- listed $2,780, fair range $3,374-$5,443

### Listings that look overpriced (actual > q90):
- listed $4,797, fair range $2,886-$4,592
- listed $25,998, fair range $13,653-$17,840
- listed $4,795, fair range $2,903-$4,468
