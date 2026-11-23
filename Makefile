.PHONY: install fetch-data train serve lint typecheck test docker-up docker-down clean help

help:
	@echo "RBP MLOps Service - Available Commands"
	@echo "======================================"
	@echo "install      : Install package and dependencies"
	@echo "fetch-data   : Download ENCODE data and reference genome"
	@echo "train        : Train RBP binding classifier"
	@echo "serve        : Start FastAPI inference server"
	@echo "lint         : Run ruff linter and formatter checks"
	@echo "typecheck    : Run mypy type checker"
	@echo "test         : Run pytest test suite"
	@echo "docker-up    : Start Docker services (MLflow + API)"
	@echo "docker-down  : Stop Docker services"
	@echo "clean        : Remove generated files and caches"

install:
	pip install -e ".[dev]"

fetch-data:
	python -m scripts.fetch_data

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
