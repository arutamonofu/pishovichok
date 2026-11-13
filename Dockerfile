# syntax=docker/dockerfile:1.7

FROM python:3.12-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

# 1. Сначала устанавливаем внешние зависимости для максимального кэширования слоёв
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev

# 2. Копируем исходники и устанавливаем сам пакет (не-editable wheel + статика UI)
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable


FROM python:3.12-slim-bookworm AS runtime

# Непривилегированный пользователь с UID/GID 1000 (совпадает со стандартным UID на Linux)
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid 1000 --create-home --shell /usr/sbin/nologin appuser

WORKDIR /app

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --chown=appuser:appuser data ./data

RUN mkdir -p /app/workspace /app/.nicegui \
    && chown -R appuser:appuser /app/workspace /app/.nicegui

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PISHOVICHOK_PROJECT_ROOT=/app \
    PISHOVICHOK_DATA_DIR=/app/data \
    PISHOVICHOK_WORKSPACE_DIR=/app/workspace \
    PISHOVICHOK_UI_HOST=0.0.0.0 \
    PISHOVICHOK_UI_PORT=8080

EXPOSE 8080

USER appuser

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/', timeout=3)"]

CMD ["pishovichok-ui"]
