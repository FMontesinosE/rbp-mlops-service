.PHONY: install fetch-data fetch-data-test fetch-data-full fetch-genome train serve lint typecheck test docker-up docker-down clean help

help:
	@echo "RBP MLOps Service - Available Commands"
	@echo "======================================"
	@echo "install          : Install package and dependencies"
	@echo "fetch-data-test  : Download test data (chr21-22 only, quick)"
	@echo "fetch-data-full  : Download FULL ENCODE eCLIP database (WARNING: large!)"
	@echo "fetch-data       : Alias for fetch-data-test"
	@echo "fetch-genome     : Download only reference genome (skip narrowPeak)"
	@echo "train            : Train RBP binding classifier"
	@echo "serve            : Start FastAPI inference server"
	@echo "lint             : Run ruff linter and formatter checks"
	@echo "typecheck        : Run mypy type checker"
	@echo "test             : Run pytest test suite"
	@echo "docker-up        : Start Docker services (MLflow + API)"
	@echo "docker-down      : Stop Docker services"
	@echo "clean            : Remove generated files and caches"

install:
	pip install -e ".[dev]"

fetch-data-test:
	python -m scripts.fetch_data --test-mode

fetch-data-full:
	@echo "WARNING: This will download the entire ENCODE eCLIP database"
	@echo "This includes all chromosomes (chr1-22, chrX, chrY) and all eCLIP experiments"
	@echo "Dataset size: ~40GB genome + variable narrowPeak files"
	@echo "Press Ctrl+C to cancel, or wait 5 seconds to proceed..."
	@sleep 5
	python -m scripts.fetch_data

fetch-data: fetch-data-test

fetch-genome:
	python -m scripts.fetch_data --chromosomes-only

train:
	rbp-train \
		--bed data/raw/rbfox2.bed \
		--fasta data/raw/GRCh38.fa \
		--epochs 50 \
		--batch-size 64 \
		--lr 0.001

serve:
	uvicorn rbp_mlops.api:app --host 0.0.0.0 --port 8000 --reload

lint:
	@echo "Running ruff check..."
	ruff check src/ tests/ scripts/
	@echo "Running ruff format check..."
	ruff format --check src/ tests/ scripts/

typecheck:
	@echo "Running mypy..."
	mypy src/

test:
	@echo "Running pytest..."
	pytest tests/ -v --cov=src/rbp_mlops --cov-report=term --cov-report=xml

docker-up:
	docker compose -f docker/docker-compose.yml up -d --build

docker-down:
	docker compose -f docker/docker-compose.yml down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
