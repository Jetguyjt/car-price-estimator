PYTHON ?= python3

.PHONY: install app data train evaluate test coverage-test coverage-smoke

install:
	$(PYTHON) -m pip install -r requirements.txt

app:
	$(PYTHON) -m streamlit run src/app.py

data:
	$(PYTHON) src/clean_raw_data.py
	$(PYTHON) src/preprocess_cars_data.py

train:
	$(PYTHON) src/train_price_model.py

evaluate:
	$(PYTHON) src/evaluate_model.py

test:
	$(PYTHON) -m pytest tests -q

coverage-test:
	$(MAKE) -C experiments/training_coverage test

coverage-smoke:
	$(MAKE) -C experiments/training_coverage smoke
