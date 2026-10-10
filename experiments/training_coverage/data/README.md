# Data

The current checkout does not redistribute the row-level vehicle advertisements used in the completed run. The records came from a third-party marketplace, and reuse remains subject to the source site's terms. Included artifacts are limited to a field schema and aggregate results. The root [`data/README.md`](../../../data/README.md) describes the boundary for older development history.

The modeling sample contained 20,801 advertisement records after cleaning and feature preparation. Prices are advertised asking prices, not completed-sale prices. Collection was performed in multiple nonrandom batches during 2026, so the sample should not be treated as representative of the broader vehicle market.

## Expected input

The analysis code expects a processed CSV table whose fields follow [`schema.csv`](schema.csv). Identifier and location fields are metadata and are excluded from model inputs. `price_per_mile` is also excluded because it contains the target. The final schema entry, `make_model_price_enc`, documents a fold-derived model input; it is created during model fitting and must not be supplied as a precomputed CSV column.

The experiment begins with this processed table. It does not include the earlier raw-to-processed collection and cleaning pipeline, which created fields such as the grouped categories, luxury indicator, and trim tier. A new run must document its own collection and preprocessing choices, prepare compatible fields, and pass the resulting table through the command-line data option described in the project README. Reproducing the saved values requires the original frozen sample; a new sample should be expected to produce different results.

Do not commit source records, processed row-level tables, fold assignments, predictions, or fitted models.
