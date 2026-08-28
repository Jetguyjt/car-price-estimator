# Quick Model Writeup

Generated: 2026-06-30 11:36:05

## Setup
- Model: lightgbm + quantile regression (q10/q50/q90) + mean point model
- Rows: total 920, train 736, test 184
- Features: 13 (6 categorical)

## Headline results
- Test R2: 0.4224  (baseline R2: -0.0732)
- Test MAE: $8,059  (baseline MAE: $12,504)
- Test RMSE: $13,787
- Test MAPE: 22.43%

## Confidence band (q10-q90)
- Coverage on test set: 70.1% (ideal ~80%)
- Average band width: $19,942

## Top features (gain)
```
       feature  gain_importance  split_importance
       mileage      1070.238483               947
          year       961.694845               409
    body_style       610.061009               118
miles_per_year       473.242231               756
is_luxury_make       363.409796               145
    make_model       246.304361                39
          trim       146.624600                53
       car_age        60.075600                24
mileage_bucket         0.413708                 1
        region         0.000000                 0
```

## Sample predictions
- Closest call: predicted $18,832, actual $18,999, error $167
- Worst miss:  predicted $28,056, actual $107,774, error $79,718

### Listings that look like deals (actual < q10):
- listed $8,497, fair range $10,649-$20,988
- listed $7,907, fair range $9,935-$25,361
- listed $13,690, fair range $15,093-$25,433

### Listings that look overpriced (actual > q90):
- listed $35,060, fair range $15,263-$27,493
- listed $38,805, fair range $19,226-$34,458
- listed $29,495, fair range $12,180-$22,120
