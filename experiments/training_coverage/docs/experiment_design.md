# Experiment design

## Question being tested

How much does the estimated error of one vehicle asking-price model change when its training data lack an exact make/model category, an entire manufacturer, a mileage range, or a vehicle-age range?

These conditions represent different prediction settings. A random listing split usually leaves related vehicles in the training set. A category holdout removes every listing in the tested category. A numerical-range holdout removes training support from a defined part of the mileage or age distribution.

## Data and prediction target

The completed run used advertisements for new and used vehicles from a major U.S. online automotive marketplace. Collection was performed in multiple nonrandom batches during 2026. The resulting records are a convenience sample shaped by marketplace availability, search choices, and collection timing; they are not a census or a simultaneous snapshot of the vehicle market.

The target is the advertised asking price in U.S. dollars. It is not a completed-sale price, appraisal, or uniquely correct market value. Each row represents an advertisement record, not necessarily a unique physical vehicle.

The source table contained 21,745 records. Primary cleaning retained 21,219 records, and fixed feature preparation produced 20,801 model-ready rows. The row-level data are not distributed in the current release. See [`data/README.md`](../data/README.md) for the expected schema and access boundary; the root data README separately describes older development history.

## Fixed prediction pipeline

Every fit used the same LightGBM regression pipeline. The model was trained on `log1p(price)` and predictions were transformed back to dollars. The fixed settings were:

- 2,000 trees
- learning rate 0.05
- 63 leaves
- minimum child size 20
- feature fraction 0.9
- bagging fraction 0.9
- bagging frequency 1
- random seed 42 for model fitting

Monotonic constraints required predicted price to be nondecreasing with model year and nonincreasing with mileage and prior-owner count, with other inputs held fixed.

The pipeline included a smoothed training-fold mean log price for each processed make/model category, with smoothing value 10. The encoding and categorical vocabularies were fitted separately inside every outer training fold. A make/model category absent from training received the training-fold overall mean. Training rows contributed to the encoding used to fit those same rows; no inner cross-fitting or leave-one-out encoding was applied.

The model used 22 inputs covering vehicle identity and trim, year and use, mechanical and body information, ownership and accident history, listing status, and derived variables. The complete field inventory appears in [`data/schema.csv`](../data/schema.csv).

## Primary evaluation conditions

Each condition produced out-of-fold predictions. Every eligible listing was scored by a model that had not trained on that listing.

1. **Random listing-level reference.** Five shuffled folds, repeated with seeds 42, 43, and 44.
2. **Processed make/model holdout.** Complete processed make/model groups assigned to five approximately row-balanced folds, repeated with seeds 42, 43, and 44.
3. **Manufacturer holdout.** Complete manufacturer groups assigned in the same way, repeated with seeds 42, 43, and 44.
4. **Mileage-range holdout.** Five fits, each excluding one complete mileage range.
5. **Vehicle-age-range holdout.** Five fits, each excluding one complete vehicle-age range.

Group allocation used a target-blind randomized greedy procedure that evaluated 500 assignments per seed and selected the most row-balanced assignment. A categorical group could not cross the train-test boundary within a repetition.

### Primary numerical ranges

| Variable | Ranges |
|---|---|
| Mileage | 0–14,999; 15,000–34,999; 35,000–59,999; 60,000–99,999; 100,000+ miles |
| Vehicle age | 0–1; 2–3; 4–5; 6–8; 9+ years |

The mileage or age value remained available to the model at prediction time. What changed was whether the corresponding region had training support.

## Matched range comparison

A separate comparison held the test listings and number of training rows constant for four ranges selected in advance:

- 35,000–59,999 miles (interior)
- 100,000+ miles (upper tail)
- ages 4–5 years (interior)
- ages 9+ years (upper tail)

For each range, the same test cohort was evaluated under two training conditions:

- **Range absent:** training used all eligible records outside the range and none inside it.
- **Range represented:** training included the other four-fifths of in-range records, then removed the same number of out-of-range records so the training count matched the range-absent condition.

When choosing out-of-range removals, the matching procedure first sought the same processed make/model, then the same manufacturer, and finally used the remaining outside pool. Content fingerprints covered every processed field except listing identifiers and location metadata. Rows with an identical fingerprint were kept on one side of the represented-condition train-test boundary. The represented construction was repeated with seeds 42, 43, and 44.

This design controls the test cohort and training count, but the two training sets are not identical in every other respect. It is a composition-aware stress test, not a causal estimate of the effect of coverage.

## Measures and uncertainty summaries

The primary measure was mean absolute error (MAE) in dollars. Secondary measures were mean absolute percentage error (MAPE), median absolute percentage error, weighted absolute percentage error, root mean squared error, and R-squared.

For repeated protocols, losses were first averaged within each listing across repetitions and then pooled. Comparisons used paired losses for the same listings. Descriptive 95% intervals for MAE differences and ratios came from 2,000 cluster-bootstrap resamples. Make/model groups were the resampling units except in the manufacturer comparison, which used manufacturer groups.

These intervals are conditional on the saved model fits and collected sample. They do not include uncertainty from collecting a different sample, retraining in another time period, or moving to another marketplace.

## Sensitivity analyses

The planned checks were:

- alternate mileage and vehicle-age boundaries;
- removal of the make/model price encoding;
- removal of mileage bucket and miles-per-year from the mileage analysis;
- removal of 25 later rows that exactly matched earlier rows on processed content after excluding listing identifiers and location metadata; and
- addition of manufacturer as an explicit model input.

Each modified pipeline was compared with its own random-reference result.

## Interpretation boundaries

The protocols measure performance under specific forms of missing training coverage. They do not establish a universal error rate, prove that an individual feature causes the error difference, evaluate completed-sale prices, or demonstrate performance in a later market or another marketplace.

Several preprocessing decisions were applied to the frozen table before cross-validation, including target-informed price and price-per-mile filters and global rare-category grouping. The numerical ranges also contain different mixtures of vehicles. Results should therefore be interpreted as an audit of this processed sample, model pipeline, and set of holdout definitions.
