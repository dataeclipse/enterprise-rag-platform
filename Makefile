.PHONY: dev test test-integration lint typecheck fmt up down eval

dev:
	uv sync --group dev --extra ml

test:
	uv run pytest tests -m "not integration" --cov=src/rag --cov-report=term-missing

test-integration:
	uv run pytest tests -m integration

lint:
	uv run ruff check src tests evals
	uv run black --check src tests evals

typecheck:
	uv run mypy

fmt:
	uv run ruff check --fix src tests evals
	uv run black src tests evals

up:
	docker compose up -d

down:
	docker compose down -v

eval:
	uv run python evals/run_retrieval_eval.py

eval-ragas:
	uv run python evals/run_ragas.py
