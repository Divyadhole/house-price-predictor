.PHONY: install data train reports lint test validate api
install:
	python3.12 -m venv .venv
	.venv/bin/python -m pip install -e '.[dev]'
data:
	.venv/bin/python scripts/generate_sample_data.py
train:
	.venv/bin/house-price train
reports:
	MPLCONFIGDIR=/tmp/house-matplotlib .venv/bin/python scripts/generate_reports.py
lint:
	.venv/bin/ruff check src tests scripts
test:
	.venv/bin/python -m pytest --cov=house_price_predictor --cov-report=term-missing
validate: lint test train reports
api:
	.venv/bin/uvicorn house_price_predictor.api:app --reload
