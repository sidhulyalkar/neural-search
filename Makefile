.PHONY: install setup dev test test-backend lint format api web demo demo-seed demo-quick demo-search clean docker-up docker-down up benchmark eval reports report notebook-generate generate-notebook build corpus-build graph-build graph-reports embeddings-build artifacts-build real-corpus-build real-claims-build real-graph-build real-embeddings-build real-reports real-artifacts-build awareness-report search-intelligence-report search-intelligence-plan human-review-queue release-check release-summary

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

# ============================================================================
# CANONICAL ARTIFACT PIPELINE
# ============================================================================

corpus-build:
	python -m neural_search.corpus.convert_demo_seed \
		--datasets data/seed/demo_datasets.yaml \
		--papers data/seed/demo_papers.yaml \
		--out-dir data/corpus/normalized

graph-build: corpus-build
	python -m neural_search.graph.build_graph \
		--datasets data/corpus/normalized/demo_v05.datasets.jsonl \
		--papers data/corpus/normalized/demo_v05.papers.jsonl \
		--out data/graph/neural_search_graph.demo_v05.json

graph-reports: graph-build
	python -m neural_search.graph.reports \
		--graph data/graph/neural_search_graph.demo_v05.json \
		--out data/reports/graph

embeddings-build: corpus-build
	python -m neural_search.embeddings.build_index \
		--input data/corpus/normalized/demo_v05.records.jsonl \
		--out data/embeddings/demo_v05.field_embeddings.jsonl \
		--provider hashing

artifacts-build: corpus-build graph-build graph-reports embeddings-build

real-corpus-build:
	python -m neural_search.corpus.ingest_manifest \
		--manifest data/corpus/manifests/real_v07.yaml \
		--out data/corpus/normalized \
		--claims-out data/corpus/claims/real_v07.claims.jsonl \
		--prefix real_v07

real-claims-build: real-corpus-build

real-graph-build: real-corpus-build
	python -m neural_search.graph.build_graph \
		--datasets data/corpus/normalized/real_v07.datasets.jsonl \
		--papers data/corpus/normalized/real_v07.papers.jsonl \
		--out data/graph/neural_search_graph.real_v07.json

real-embeddings-build: real-corpus-build
	python -m neural_search.embeddings.build_index \
		--input data/corpus/normalized/real_v07.records.jsonl \
		--out data/embeddings/real_v07.field_embeddings.jsonl \
		--provider hashing

real-reports: real-corpus-build real-graph-build
	python -m neural_search.graph.reports \
		--graph data/graph/neural_search_graph.real_v07.json \
		--out data/reports/real_v07/graph
	python -m neural_search.corpus.real_reports \
		--manifest data/corpus/manifests/real_v07.yaml \
		--records data/corpus/normalized/real_v07.records.jsonl \
		--claims data/corpus/claims/real_v07.claims.jsonl \
		--out data/reports/real_v07

real-artifacts-build: real-corpus-build real-claims-build real-graph-build real-embeddings-build real-reports

awareness-report: corpus-build real-corpus-build
	python -m neural_search.awareness.report \
		--records data/corpus/normalized \
		--out data/reports/awareness

search-intelligence-report: corpus-build
	python -m neural_search.intelligence.coverage \
		--records data/corpus/normalized \
		--benchmark data/eval/benchmark_queries_real_v07.yaml \
		--out data/reports/search_intelligence

search-intelligence-plan:
	@if [ -z "$(QUERY)" ]; then \
		echo "Usage: make search-intelligence-plan QUERY='<query>'"; \
	else \
		python -m neural_search.intelligence.planner \
			--query "$(QUERY)" \
			--corpus-profile data/reports/search_intelligence/search_coverage_plan.json; \
	fi

human-review-queue: search-intelligence-report
	python -m neural_search.intelligence.review \
		--coverage data/reports/search_intelligence/search_coverage_plan.json \
		--benchmark-seeds data/reports/search_intelligence/benchmark_query_seeds.yaml \
		--out data/reports/search_intelligence

release-summary:
	python -m neural_search.release.check --summary-only

release-check:
	pytest -q
	ruff check neural_search tests
	$(MAKE) artifacts-build
	$(MAKE) real-artifacts-build
	python -m neural_search.evaluation.run_benchmark --suite demo_v02
	python -m neural_search.evaluation.run_benchmark --suite adversarial
	python -m neural_search.evaluation.run_benchmark --suite real_v07
	python -m neural_search.release.check

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
# CORPUS EXPANSION
# ============================================================================

# Run corpus expansion phase 1 (high priority)
corpus-expand-phase1:
	@echo "Running Corpus Expansion Phase 1..."
	bash scripts/run_corpus_expansion.sh 1

# Run corpus expansion phase 2 (medium priority)
corpus-expand-phase2:
	@echo "Running Corpus Expansion Phase 2..."
	bash scripts/run_corpus_expansion.sh 2

# Run corpus expansion phase 3 (gap filling)
corpus-expand-phase3:
	@echo "Running Corpus Expansion Phase 3..."
	bash scripts/run_corpus_expansion.sh 3

# Run all corpus expansion phases
corpus-expand-all:
	@echo "Running All Corpus Expansion Phases..."
	bash scripts/run_corpus_expansion.sh all

# Show corpus status
corpus-status:
	@echo "=== Corpus Status ==="
	@echo "DANDI datasets: $$(ls data/raw/dandi/*.json 2>/dev/null | wc -l) raw files"
	@echo "OpenNeuro datasets: $$(ls data/raw/openneuro/*.json 2>/dev/null | wc -l) raw files"
	@echo "OpenAlex papers: $$(ls data/raw/openalex/*.json 2>/dev/null | wc -l) raw files"
	@echo ""
	@echo "Log files: $$(ls data/logs/ingestion/*.log 2>/dev/null | wc -l)"
	@echo "Failures: $$(cat data/logs/ingestion/failures_*.csv 2>/dev/null | wc -l)"

# Generate corpus coverage report
corpus-coverage:
	python -m neural_search.reports.coverage

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
