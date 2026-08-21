# CarPrice - reproducible pipeline.
# Every result in docs/paper_results.md regenerates from the committed processed
# dataset with `make experiments`. Targets run from the repo root.

PY := python3

.PHONY: help install test data train evaluate experiments all clean-pyc

help:
	@echo "Targets:"
	@echo "  install      pip install -r requirements.txt"
	@echo "  data         raw -> interim -> processed (re-clean + re-feature-engineer)"
	@echo "  train        train + calibrate + version the model (writes model_artifacts/)"
	@echo "  evaluate     baselines, segment errors, calibration curve, SHAP"
	@echo "  experiments  run every paper study -> docs/paper_results.md + figures"
	@echo "  test         run the unit-test suite (pytest)"
	@echo "  all          data + train + evaluate + experiments + test"

install:
	$(PY) -m pip install -r requirements.txt

# Data pipeline (needs data/raw/*.csv or the SQLite export present).
data:
	$(PY) src/clean_raw_data.py
	$(PY) src/preprocess_cars_data.py

train:
	$(PY) src/train_price_model.py

evaluate:
	cd src && $(PY) evaluate_model.py

# Reproduces the entire results section (retrains models internally; ~10-15 min).
experiments:
	cd src && $(PY) -m experiments.run_all

test:
	$(PY) -m pytest tests/ -q

all: data train evaluate experiments test

clean-pyc:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
