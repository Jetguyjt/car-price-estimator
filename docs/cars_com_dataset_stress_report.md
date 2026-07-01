# Cars.com raw dataset stress report for price prediction

CSV: `data/raw/cars_com_raw.csv`
_Point-in-time report generated on a ~1,000-row snapshot; regenerate with `python3 src/stress_test_cars_dataset.py`._
Total rows: **1022**; unique listing keys: **1022**; duplicate keys: **0**
Rows with detail URLs: **471**; rows without listing_url, likely older/card-only rows: **551**

## Target/coverage
Rows with usable positive price: **1014 / 1022 (99.2%)**
Rows usable for basic ML (price + year + make + model): **1005 / 1022 (98.3%)**
Realistic filtered ML rows ($1k-$250k, years 1980-2027): **996 / 1022 (97.5%)**

## Missingness worst fields
- `mpg_city`: 1022 missing (100.0%)
- `mpg_highway`: 1022 missing (100.0%)
- `mpg_combined`: 1022 missing (100.0%)
- `mpg_equivalent`: 1022 missing (100.0%)
- `no_accidents_raw`: 1021 missing (99.9%)
- `owner_count`: 1006 missing (98.4%)
- `title_status_raw`: 1000 missing (97.8%)
- `mpg_text`: 958 missing (93.7%)
- `zip`: 858 missing (84.0%)
- `region`: 858 missing (84.0%)
- `fuel_type_raw`: 638 missing (62.4%)
- `drivetrain_raw`: 621 missing (60.8%)
- `interior_color`: 589 missing (57.6%)
- `engine_raw`: 582 missing (56.9%)
- `exterior_color`: 580 missing (56.8%)
- `transmission_raw`: 579 missing (56.7%)
- `stock_number`: 576 missing (56.4%)
- `cpo_raw`: 575 missing (56.3%)
- `enriched_at`: 575 missing (56.3%)
- `listing_url`: 551 missing (53.9%)

## Sanity flags
- missing_price: **16**
- missing_year: **8**
- missing_mileage: **8**
- implausible_year: **0**
- used_or_cpo_low_mileage_lt100: **1**
- new_high_mileage_gt1000: **41**
- used_price_lt1000: **6**
- price_gt250k: **4**
- title_stock_mismatch: **24**
- cpo_mismatch: **24**

## Stock type counts
```json
{
  "Used": 868,
  "Certified": 76,
  "New": 70,
  "(blank)": 8
}
```
## Top makes
```json
{
  "Toyota": 104,
  "Nissan": 100,
  "Ford": 79,
  "Subaru": 73,
  "Chevrolet": 69,
  "Hyundai": 65,
  "Honda": 63,
  "Kia": 41,
  "BMW": 40,
  "Jeep": 34,
  "Volkswagen": 31,
  "Volvo": 30,
  "Audi": 26,
  "Mazda": 25,
  "Mercedes-Benz": 23,
  "Lexus": 22,
  "GMC": 21,
  "(blank)": 17,
  "Cadillac": 16,
  "Dodge": 15,
  "Acura": 15,
  "Mitsubishi": 13,
  "Chrysler": 13,
  "Tesla": 13,
  "INFINITI": 11
}
```
## Price distribution
```json
{
  "count": 1014.0,
  "mean": 32298.97,
  "std": 38452.07,
  "min": 180.0,
  "1%": 1945.41,
  "5%": 4999.0,
  "10%": 9699.3,
  "25%": 16998.0,
  "50%": 24998.0,
  "75%": 36998.0,
  "90%": 51998.0,
  "95%": 73056.8,
  "99%": 159693.91,
  "max": 699995.0
}
```
## Mileage distribution
```json
{
  "count": 1014.0,
  "mean": 51704.51,
  "std": 46506.07,
  "min": 0.0,
  "1%": 4.0,
  "5%": 1772.3,
  "10%": 3217.9,
  "25%": 17646.75,
  "50%": 41361.5,
  "75%": 71398.0,
  "90%": 112367.3,
  "95%": 143162.35,
  "99%": 209447.37,
  "max": 271306.0
}
```
## Most repeated exact titles
```json
{
  "": 8,
  "Used 2023 Nissan Rogue SV": 6,
  "Used 2024 Toyota Tacoma SR5": 4,
  "Used 2024 Hyundai ELANTRA SEL": 3,
  "Used 2019 Ford Ranger XLT": 3,
  "Used 2023 Hyundai TUCSON SEL": 3,
  "Used 2020 Hyundai KONA SEL": 3,
  "Used 2025 Honda CR-V EX-L": 3,
  "Used 2023 Tesla Model 3": 3,
  "Used 2025 Toyota 4Runner SR5": 3,
  "Used 2023 GMC Sierra 1500 Elevation": 3,
  "Certified 2024 Subaru Crosstrek Premium": 3,
  "Certified 2024 Subaru Forester Wilderness": 3,
  "Used 2020 Nissan Kicks SR": 2,
  "Used 2024 Ford Edge SEL": 2,
  "Used 2022 Audi Q5 45 S line Premium": 2,
  "Used 2025 Volkswagen Jetta 1.5T S": 2,
  "Used 2024 Audi A4 45 S line Premium Plus": 2,
  "Used 2023 Volvo C40 Recharge Pure Electric Twin Plus": 2,
  "Used 2020 Nissan Sentra SV": 2
}
```

## Baseline model stress tests
| split | model | n_train | n_test | mae | medae | mape | r2 | within_10pct | within_20pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| holdout_make_Chevrolet | rf_log | 928 | 68 | 9377.569 | 6008.024 | 0.985 | 0.493 | 0.162 | 0.294 |
| holdout_make_Chevrolet | ridge_log | 928 | 68 | 12655.907 | 8424.341 | 1.570 | 0.088 | 0.132 | 0.221 |
| holdout_make_Chevrolet | dummy_median | 928 | 68 | 14025.132 | 10001.500 | 1.650 | -0.018 | 0.074 | 0.221 |
| holdout_make_Ford | ridge_log | 917 | 79 | 8524.371 | 4987.425 | 0.655 | 0.673 | 0.215 | 0.456 |
| holdout_make_Ford | rf_log | 917 | 79 | 11072.547 | 6142.262 | 0.516 | 0.383 | 0.190 | 0.329 |
| holdout_make_Ford | dummy_median | 917 | 79 | 18307.570 | 14879.000 | 0.970 | -0.101 | 0.038 | 0.127 |
| holdout_make_Honda | ridge_log | 933 | 63 | 3801.766 | 2559.520 | 0.200 | 0.764 | 0.365 | 0.635 |
| holdout_make_Honda | rf_log | 933 | 63 | 4571.681 | 3026.303 | 0.236 | 0.656 | 0.349 | 0.540 |
| holdout_make_Honda | dummy_median | 933 | 63 | 8294.571 | 6000.000 | 0.963 | -0.008 | 0.238 | 0.460 |
| holdout_make_Nissan | rf_log | 896 | 100 | 7909.737 | 6570.242 | 0.389 | -0.503 | 0.210 | 0.350 |
| holdout_make_Nissan | ridge_log | 896 | 100 | 8595.232 | 8193.640 | 0.414 | -0.263 | 0.130 | 0.270 |
| holdout_make_Nissan | dummy_median | 896 | 100 | 9098.770 | 8000.000 | 0.875 | -0.265 | 0.100 | 0.250 |
| holdout_make_Subaru | dummy_median | 926 | 70 | 7657.357 | 6000.000 | 0.389 | -0.114 | 0.200 | 0.429 |
| holdout_make_Subaru | ridge_log | 926 | 70 | 9577.314 | 8004.673 | 0.341 | -0.762 | 0.186 | 0.343 |
| holdout_make_Subaru | rf_log | 926 | 70 | 17347.357 | 9832.464 | 0.598 | -5.673 | 0.186 | 0.329 |
| holdout_make_Toyota | ridge_log | 893 | 103 | 6957.195 | 4915.741 | 0.227 | 0.463 | 0.204 | 0.524 |
| holdout_make_Toyota | rf_log | 893 | 103 | 7031.393 | 5177.467 | 0.230 | 0.460 | 0.282 | 0.515 |
| holdout_make_Toyota | dummy_median | 893 | 103 | 11561.728 | 9000.000 | 0.448 | -0.237 | 0.175 | 0.359 |
| random_stratified_stock | ridge_log | 747 | 249 | 5286.023 | 2679.322 | 0.324 | 0.674 | 0.438 | 0.699 |
| random_stratified_stock | rf_log | 747 | 249 | 7241.283 | 3652.456 | 0.319 | 0.520 | 0.293 | 0.554 |
| random_stratified_stock | dummy_median | 747 | 249 | 13045.944 | 9001.000 | 0.833 | -0.011 | 0.153 | 0.297 |
| train_2018plus_test_pre2018 | ridge_log | 762 | 234 | 4558.729 | 2804.778 | 0.524 | 0.211 | 0.256 | 0.479 |
| train_2018plus_test_pre2018 | rf_log | 762 | 234 | 6883.900 | 5413.265 | 0.980 | -0.211 | 0.175 | 0.303 |
| train_2018plus_test_pre2018 | dummy_median | 762 | 234 | 16759.731 | 15527.000 | 2.495 | -2.437 | 0.026 | 0.056 |

## Notes
- Metrics are intentionally rough stress tests, not final model claims.
- Target was trained as log(price), then predictions were transformed back to dollars.
- Random split likely overstates real performance because similar vehicles/dealers/price bands appear in train and test.
- Holdout-make and old-vehicle tests are more realistic stress tests for generalization.