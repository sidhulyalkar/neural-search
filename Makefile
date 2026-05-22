.PHONY: install dev test test-backend lint format api web demo demo-seed clean docker-up docker-down

# Install dependencies
install:
	pip install -e ".[all]"
	cd apps/web && npm install

# Development setup
dev:
	pip install -e ".[dev]"

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

# Run API server
api:
	uvicorn apps.api.main:app --reload --port 8000

# Run web frontend
web:
	cd apps/web && npm run dev

# Run demo script
demo:
	python scripts/demo.py

# Run full stack locally
run:
	docker-compose up -d postgres redis
	make api &
	make web

# Docker
docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-build:
	docker-compose build

# Database
db-migrate:
	alembic upgrade head

db-revision:
	alembic revision --autogenerate -m "$(msg)"

# Seed data
seed:
	python scripts/seed_demo_data.py

demo-seed:
	python -m neural_search.ingestion.demo_seed

# Clean
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf dist/ build/ *.egg-info/
