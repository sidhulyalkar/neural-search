.PHONY: install dev test lint format api web demo clean docker-up docker-down

# Install dependencies
install:
	pip install -e ".[all]"
	cd apps/web && npm install

# Development setup
dev:
	pip install -e ".[dev]"

# Run tests
test:
	pytest tests/ -v --cov=src --cov-report=term-missing

test-unit:
	pytest tests/unit -v

test-integration:
	pytest tests/integration -v

# Linting and formatting
lint:
	ruff check .
	mypy src/

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

# Clean
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf dist/ build/ *.egg-info/
