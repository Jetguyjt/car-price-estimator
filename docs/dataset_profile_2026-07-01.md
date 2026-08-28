# Dataset profile - Cars.com raw listings

Generated from `data/raw/cars_com_raw.csv` on 2026-07-01.

## Summary

| Metric | Value |
|---|---:|
| Rows | 2,668 |
| Unique listing keys | 2,668 |
| Duplicate listing keys | 0 |
| Rows with non-empty price | 2,660 |
| Overall median price | $4,999 |
| Used-listing median price | $4,995 |
| Overall mean price | $16,003 |

## Stock-type mix

| Stock type | Rows |
|---|---:|
| Used | 2,508 |
| Certified | 82 |
| New | 70 |
| Blank/unknown | 8 |

## Price distribution

| Price band | All rows | Used rows |
|---|---:|---:|
| $0k-$5k | 1,359 | 1,348 |
| $5k-$10k | 174 | 164 |
| $10k-$15k | 199 | 185 |
| $15k-$20k | 194 | 187 |
| $20k-$25k | 208 | 194 |
| $25k-$30k | 158 | 140 |
| $30k-$40k | 165 | 144 |
| $40k-$50k | 94 | 80 |
| $50k-$70k | 53 | 42 |
| $70k-$100k | 24 | 13 |
| $100k-$150k | 19 | 8 |
| $150k+ | 13 | 3 |

## Percentiles

| Percentile | All prices | Used prices |
|---|---:|---:|
| p05 | $2,495 | $2,495 |
| p10 | $2,999 | $2,999 |
| p25 | $3,999 | $3,995 |
| p50 | $4,999 | $4,995 |
| p75 | $21,998 | $19,998 |
| p90 | $36,531 | $32,998 |
| p95 | $46,023 | $41,998 |

## Make concentration

Top makes by row count:

| Make | Rows |
|---|---:|
| Ford | 295 |
| Nissan | 286 |
| Chevrolet | 264 |
| Toyota | 212 |
| Honda | 160 |
| Hyundai | 145 |
| Subaru | 138 |
| Kia | 127 |
| Jeep | 102 |
| Volkswagen | 82 |
| BMW | 79 |
| Dodge | 78 |
| Chrysler | 60 |
| Mazda | 59 |
| GMC | 55 |
| Mercedes-Benz | 50 |
| Volvo | 47 |
| Lexus | 41 |
| Audi | 40 |
| Buick | 40 |

## Missingness highlights

| Field | Missing rows | Missing % |
|---|---:|---:|
| price | 8 | 0.3% |
| mileage | 8 | 0.3% |
| year | 8 | 0.3% |
| make | 29 | 1.1% |
| model | 29 | 1.1% |
| body_style | 113 | 4.2% |
| stock_number | 757 | 28.4% |
| exterior_color | 796 | 29.8% |
| interior_color | 883 | 33.1% |
| fuel_type_raw | 835 | 31.3% |
| drivetrain_raw | 811 | 30.4% |
| transmission_raw | 764 | 28.6% |
| engine_raw | 763 | 28.6% |
| mpg_text | 2,407 | 90.2% |
| title_status_raw | 2,460 | 92.2% |
| no_accidents_raw | 2,666 | 99.9% |
| owner_count | 2,597 | 97.3% |
| enriched_at | 575 | 21.6% |

## Main diagnosis

The raw dataset is currently **heavily skewed toward very cheap vehicles**. More than half of priced rows are below $5k, so the median is about $5k. That is not representative of the mainstream used-car market and will pull any model toward low-price inventory unless the training set is filtered, weighted, or balanced.

The current data is useful as a seed dataset and for parser/model pipeline development, but it is not yet balanced enough for a portfolio-ready price-prediction model.

## What needs to improve next

1. **Balance price coverage.** Prioritize collection in the $20k-$50k range. To move the raw median above $20k without deleting rows, roughly 1.2k additional rows priced at/above $20k are needed. To move the median closer to $25k, roughly 1.6k additional rows above $25k are needed. To approach $30k, roughly 1.9k additional rows above $30k are needed.
2. **Do not delete raw cheap rows.** Keep raw data intact for provenance. Create a balanced training view instead of rewriting history.
3. **Create a balanced modeling dataset.** Downsample the <$10k rows or use sample weights so training is not dominated by older cheap vehicles.
4. **Improve detail enrichment coverage.** Core detail fields are present for about 67-71% of rows, but history fields such as accidents, owner count, and title status are mostly missing.
5. **Validate price parsing.** There are a few suspicious prices such as very low values in non-used rows. New/CPO rows should be excluded or revalidated before training.
6. **Re-run evaluation after balancing.** Current metrics were trained on an earlier snapshot and should be regenerated after a balanced dataset is built.

## Recommended next collection target

For the next scrape, skip broad cheap searches and focus on searches likely to produce rows in the model's target market:

- Stock: `used` first; add CPO later only after price parsing is validated.
- Price floor: prioritize $20k+ inventory.
- Preferred target bands: $20k-$25k, $25k-$30k, $30k-$40k, $40k-$50k.
- Makes: Toyota, Honda, Ford, Chevrolet, Nissan, Hyundai, Kia, Subaru, Jeep, Volkswagen, Mazda, BMW, Lexus, Audi, Mercedes-Benz, GMC.

The goal should be to get each of the $20k-$50k bands to at least several hundred rows before collecting more under-$10k inventory.
