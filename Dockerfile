# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# Optional provider SDKs: docker build --build-arg EXTRAS=vertex,openai .
ARG EXTRAS=vertex,openai
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install ".[${EXTRAS}]"

COPY prompts ./prompts
COPY data ./data
COPY eval ./eval

RUN useradd --create-home --uid 10001 app && mkdir -p /app/.cache && chown -R app /app
USER app

ENV RAG_PROMPTS_DIR=/app/prompts \
    RAG_CATALOGUE_CSV=/app/data/raw/retail_toy_dataset.csv \
    RAG_MEMORY_STORE_PATH=/app/.cache/memory_index \
    PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/health', timeout=4)"
# One worker per container: scale out with replicas/instances. (Prometheus counters are per process;
# with several workers each would report its own numbers.)
CMD ["sh", "-c", "uvicorn retail_rag.api.app:create_app --factory --host 0.0.0.0 --port ${PORT} --workers ${WEB_CONCURRENCY:-1}"]
