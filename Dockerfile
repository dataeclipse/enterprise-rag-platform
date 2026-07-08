FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
ARG WITH_ML=false
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$WITH_ML" = "true" ]; then \
        uv sync --frozen --no-install-project --no-dev --extra ml; \
    else \
        uv sync --frozen --no-install-project --no-dev; \
    fi
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$WITH_ML" = "true" ]; then \
        uv sync --frozen --no-dev --extra ml; \
    else \
        uv sync --frozen --no-dev; \
    fi

FROM python:3.12-slim-bookworm
WORKDIR /app
RUN groupadd --system app && useradd --system --gid app app
COPY --from=builder /app/.venv /app/.venv
COPY src ./src
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1
USER app
EXPOSE 8000
CMD ["uvicorn", "rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
