# Quick Model Writeup

Generated: 2026-07-13 21:20:31

## Setup
- Model: lightgbm + quantile regression (q10/q50/q90) + mean point model
- Rows: total 12739, train 10191, test 2548
- Features: 23 (11 categorical)

## Headline results
- Test R2: 0.8540  (baseline R2: -0.0221)
- Test MAE: $2,947  (baseline MAE: $11,836)
- Test RMSE: $7,481
- Test MAPE: 12.61%

## Confidence band (q10-q90)
- Coverage on test set: 80.4% (ideal ~80%)
- Average band width: $9,960

## Top features (gain)
```
             feature  gain_importance  split_importance
             mileage     82528.265624             53309
      miles_per_year     72200.375583             54566
          make_model     62252.324594             17755
                year     51693.980487             20337
make_model_price_enc     36108.142426             22861
                trim     35099.531865             12445
       engine_size_l     29264.935089             22279
                 mpg     19047.931520             14494
           car_color     12962.906902              9378
   transmission_type      6989.007580              5503
```

## Sample predictions
- Closest call: predicted $20,997, actual $20,998, error $1
- Worst miss:  predicted $58,323, actual $239,991, error $181,668

### Listings that look like deals (actual < q10):
- listed $4,357, fair range $4,644-$7,339
- listed $3,495, fair range $3,744-$9,684
- listed $18,925, fair range $19,010-$27,614

### Listings that look overpriced (actual > q90):
- listed $48,994, fair range $32,499-$48,646
- listed $33,500, fair range $4,185-$15,953
- listed $17,596, fair range $12,235-$16,872
