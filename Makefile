.DEFAULT_GOAL := help

.PHONY: help venv install-dev pre-commit-install \
        lint format format-check typecheck check \
        test train tune evaluate api \
        migrate seed up down \
        clean

help: ## Show this list of targets
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

venv: ## Create .venv (Python 3.12) and install runtime dependencies
	python3.12 -m venv .venv
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install -e . --no-deps

install-dev: ## Install dev tooling (ruff, mypy, pre-commit) on top of venv
	pip install -r requirements-dev.txt

pre-commit-install: ## Register the pre-commit git hook (run once per clone)
	pre-commit install

lint: ## Ruff lint check (see CI/CD section, README)
	ruff check .

format: ## Auto-format with ruff
	ruff format .

format-check: ## Check formatting without changing files (what CI runs)
	ruff format --check .

typecheck: ## mypy against src/
	mypy src/

check: lint format-check typecheck ## Run everything CI's lint job runs, locally

test: ## Run the test suite (needs `createdb credit_risk_test -O credit_risk` once - see Tests section)
	pytest

train: ## Run CV for all 8 baseline models, logging to MLflow
	python -m ml.training.train_baseline

tune: ## Run Optuna hyperparameter tuning for all 8 models
	python -m ml.training.tune

evaluate: ## Save tuned artifacts + the one-time test-set evaluation (see Final Evaluation section)
	python -m ml.evaluation.final_evaluation

api: ## Run the FastAPI dev server with reload (see API section)
	python -m api.main

migrate: ## Apply Alembic migrations to DATABASE_URL
	alembic upgrade head

seed: ## Populate the models table from db/seed_data.json
	python -m db.seed_models

up: ## Build and start the full stack (API + Postgres) via Docker Compose
	docker compose up --build

down: ## Stop the Docker Compose stack
	docker compose down

clean: ## Remove Python/pytest caches
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +
	rm -rf .pytest_cache
