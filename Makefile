.PHONY: install setup dev test test-backend lint format api web demo demo-seed demo-quick demo-search clean docker-up docker-down up benchmark eval reports report notebook-generate generate-notebook build

# ============================================================================
# SETUP TARGETS
# ============================================================================

# Install dependencies
install:
	pip install -e ".[all]"
	cd apps/web && npm install

# Alias for install
setup: install

# Development setup
dev:
	pip install -e ".[dev]"

# ============================================================================
# BUILD & TEST TARGETS
# ============================================================================

# Run tests
test:
	pytest tests/ -v --cov=neural_search --cov-report=term-missing

test-backend:
	pytest tests/ -q

test-unit:
	pytest tests/unit -v

test-integration:
	pytest tests/integration -v

# Linting and formatting
lint:
	ruff check .
	mypy neural_search/

format:
	ruff check --fix .
	ruff format .

# Build frontend
build:
	cd apps/web && npm run build

# ============================================================================
# SERVER TARGETS
# ============================================================================

# Run API server
api:
	uvicorn apps.api.main:app --reload --port 8000

# Run web frontend
web:
	cd apps/web && npm run dev

# Run full stack locally
run:
	docker-compose up -d postgres redis
	make api &
	make web

# Alias for run
up: run

# Docker
docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-build:
	docker-compose build

# ============================================================================
# DEMO TARGETS
# ============================================================================

# Run full demo (recommended entry point)
demo:
	@echo ""
	@echo "Running Neural Search Demo..."
	@echo ""
	python scripts/demo.py

# Quick demo - just seed and search
demo-quick:
	@echo "Quick demo: seeding database and running sample search..."
	python -m neural_search.ingestion.demo_seed
	@echo ""
	@echo "Running sample search..."
	python -c "from neural_search.search import search_datasets; from neural_search.ingestion.demo_seed import build_demo_seed; \
		data = build_demo_seed(); \
		r = search_datasets('reversal learning reward omission', {}, data, limit=3); \
		print(f'Found {len(r.results)} results:'); \
		[print(f'  - {x.dataset_id}: {x.score:.1f}') for x in r.results]"

# Seed demo database only
demo-seed:
	python -m neural_search.ingestion.demo_seed

# Run demo search with custom query
demo-search:
	@if [ -z "$(QUERY)" ]; then \
		python -m neural_search.search.run "Find reversal learning datasets with reward omission"; \
	else \
		python -m neural_search.search.run "$(QUERY)"; \
	fi

# ============================================================================
# EVALUATION & REPORTING
# ============================================================================

# Run benchmark evaluation
benchmark:
	@if [ -f data/eval/results/baseline_before_retrieval_upgrade/latest_eval_report.json ]; then \
		python -m neural_search.evaluation.run_benchmark \
			--compare-to data/eval/results/baseline_before_retrieval_upgrade/latest_eval_report.json; \
	else \
		python -m neural_search.evaluation.run_benchmark; \
	fi

# Alias for benchmark
eval: benchmark

# Generate compilation report
reports:
	python -m neural_search.reports

# Alias for reports
report: reports

# Generate notebook for a dataset
notebook-generate:
	@if [ -z "$(DATASET_ID)" ]; then \
		echo "Usage: make notebook-generate DATASET_ID=<id>"; \
		echo ""; \
		echo "Available datasets:"; \
		python -m neural_search.notebooks.generate --list; \
	else \
		python -m neural_search.notebooks.generate --dataset-id $(DATASET_ID); \
	fi

# Alias for notebook-generate
generate-notebook: notebook-generate

# ============================================================================
# DATABASE TARGETS
# ============================================================================

# Database migrations
db-migrate:
	alembic upgrade head

db-revision:
	alembic revision --autogenerate -m "$(msg)"

# Seed data (alias)
seed: demo-seed

# ============================================================================
# CLEANUP
# ============================================================================

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf dist/ build/ *.egg-info/

clean-data:
	rm -rf data/seed/demo_seed.db
	rm -rf data/notebooks/generated/
	rm -rf data/eval/results/
	rm -rf data/reports/

# ============================================================================
# HELP
# ============================================================================

help:
	@echo "Neural Search Makefile"
	@echo ""
	@echo "SETUP:"
	@echo "  make setup             Install all dependencies (alias: install)"
	@echo "  make dev               Install dev dependencies only"
	@echo ""
	@echo "DEMO:"
	@echo "  make demo              Run full demo with all steps"
	@echo "  make demo-quick        Quick demo (seed + sample search)"
	@echo "  make demo-seed         Seed demo database only"
	@echo "  make demo-search       Run demo search (use QUERY=\"...\" for custom)"
	@echo "  make eval              Run benchmark evaluation (alias: benchmark)"
	@echo "  make report            Generate compilation reports (alias: reports)"
	@echo "  make generate-notebook Generate notebook (use DATASET_ID=<id>)"
	@echo ""
	@echo "DEVELOPMENT:"
	@echo "  make api               Run API server on :8000"
	@echo "  make web               Run frontend on :5173"
	@echo "  make up                Run full stack (alias: run)"
	@echo "  make build             Build frontend for production"
	@echo ""
	@echo "TESTING:"
	@echo "  make test              Run all tests with coverage"
	@echo "  make test-backend      Run backend tests (quick)"
	@echo "  make lint              Run linters"
	@echo "  make format            Auto-format code"
	@echo ""
	@echo "CLEANUP:"
	@echo "  make clean             Clean build artifacts"
	@echo "  make clean-data        Clean generated data files"
