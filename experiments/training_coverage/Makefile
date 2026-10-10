PYTHON ?= python3
RUN_DIR ?= outputs/reproduced
RAW_DATA ?= ../../data/raw/vehicle_listings_raw.csv
CLEAN_DATA ?= ../../data/interim/vehicle_listings_clean.csv
PROCESSED_DATA ?= ../../data/processed/vehicle_listings_processed.csv
PUBLISH_DIR ?= build/published-results
FROZEN_RUN_MANIFEST ?=
FROZEN_RUN_MANIFEST_ARG = $(if $(strip $(FROZEN_RUN_MANIFEST)),--frozen-run-manifest "$(FROZEN_RUN_MANIFEST)",)

.PHONY: install test smoke publish-results

install:
	$(PYTHON) -m pip install -e .
	$(PYTHON) -m pip install -r requirements-dev.txt

test:
	PYTHONPATH=src $(PYTHON) -m pytest

smoke:
	PYTHONPATH=src $(PYTHON) -m coverage_shift --synthetic-smoke --output-dir build/smoke --reference-year 2026 --force

publish-results:
	$(PYTHON) scripts/publish_results.py \
		--run-dir "$(RUN_DIR)" \
		--raw-data "$(RAW_DATA)" \
		--clean-data "$(CLEAN_DATA)" \
		--processed-data "$(PROCESSED_DATA)" \
		--output-dir "$(PUBLISH_DIR)" $(FROZEN_RUN_MANIFEST_ARG)
