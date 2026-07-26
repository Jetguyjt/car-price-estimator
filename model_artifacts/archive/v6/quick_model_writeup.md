# Quick Model Writeup

Generated: 2026-07-13 21:15:46

## Setup
- Model: lightgbm + quantile regression (q10/q50/q90) + mean point model
- Rows: total 12739, train 10191, test 2548
- Features: 22 (11 categorical)

## Headline results
- Test R2: 0.8372  (baseline R2: -0.0221)
- Test MAE: $3,176  (baseline MAE: $11,836)
- Test RMSE: $7,898
- Test MAPE: 13.06%

## Confidence band (q10–q90)
- Coverage on test set: 81.0% (ideal ~80%)
- Average band width: $10,699

## Top features (gain)
```
          feature  gain_importance  split_importance
       make_model     46500.104902             14127
          mileage     38899.907746             22540
   miles_per_year     32726.173399             23833
             year     26061.790797              9520
             trim     23815.254705              9975
    engine_size_l     13468.234053              9710
        car_color      9545.432439              6907
              mpg      7481.328665              5561
       drivetrain      2734.913638              1622
transmission_type      2690.107386              2064
```

## Sample predictions
- Closest call: predicted $18,997, actual $18,998, error $1
- Worst miss:  predicted $67,112, actual $239,991, error $172,879

### Listings that look like deals (actual < q10):
- listed $4,357, fair range $4,672–$6,882
- listed $3,495, fair range $3,748–$9,581
- listed $15,777, fair range $16,454–$20,801

### Listings that look overpriced (actual > q90):
- listed $33,500, fair range $3,160–$10,640
- listed $45,998, fair range $23,938–$38,776
- listed $37,407, fair range $25,885–$34,814
