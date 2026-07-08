.PHONY: dev test test-integration lint typecheck fmt up down eval

dev:
	uv sync --group dev --extra ml

test:
	uv run pytest tests -m "not integration" --cov=src/rag --cov-report=term-missing

test-integration:
	uv run pytest tests -m integration

lint:
	uv run ruff check src tests
	uv run black --check src tests

typecheck:
	uv run mypy

fmt:
	uv run ruff check --fix src tests
	uv run black src tests

up:
	docker compose up -d

down:
	docker compose down -v

eval:
	uv run python evals/run_ragas.py
