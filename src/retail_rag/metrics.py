"""Prometheus metrics (scraped from GET /metrics)."""

from prometheus_client import Counter, Histogram

REQUESTS = Counter("rag_http_requests_total", "HTTP requests", ["route", "status"])
LATENCY = Histogram(
    "rag_http_request_seconds", "HTTP latency", ["route"], buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30)
)
STAGE = Histogram(
    "rag_stage_seconds", "RAG stage latency", ["stage"], buckets=(0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10)
)
OUTCOMES = Counter("rag_answers_total", "Answer outcomes", ["outcome"])  # answered | refused | ungrounded
TOKENS = Counter("rag_llm_tokens_total", "LLM tokens", ["direction"])
FLAGS = Counter("rag_input_flags_total", "Input screening flags", ["flag"])
