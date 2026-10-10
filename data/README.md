# Local data layout

The row-level listing data used to train and evaluate the saved model are not included in the current checkout. They came from a third-party marketplace and may contain source URLs and other fields that should not be redistributed. Earlier development commits contain old marketplace-data snapshots and source URLs; those historical files are not inputs to the current release.

The main pipeline uses these local paths:

```text
data/raw/vehicle_listings_raw.csv
data/interim/vehicle_listings_clean.csv
data/processed/vehicle_listings_processed.csv
```

`clean_raw_data.py` reads the raw file and writes the interim file. `preprocess_cars_data.py` reads the interim file and writes the processed file used by `train_price_model.py` and `evaluate_model.py`.

The processed schema used by the training-coverage experiments is listed in [`../experiments/training_coverage/data/schema.csv`](../experiments/training_coverage/data/schema.csv). A separately obtained dataset can be adapted to that schema, but it will not reproduce the saved numbers unless it contains the same records and preprocessing decisions.

All row-level data folders are ignored by Git at the current tip.
