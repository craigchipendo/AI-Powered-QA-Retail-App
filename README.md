# AI-Powered Retail Q&A: A Production-Style RAG Service

[![CI](https://github.com/craigchipendo/AI-Powered-QA-Retail-App/actions/workflows/ci.yml/badge.svg)](https://github.com/craigchipendo/AI-Powered-QA-Retail-App/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![Postgres](https://img.shields.io/badge/Postgres-pgvector-336791)
![API](https://img.shields.io/badge/API-FastAPI-009688)

A retrieval-augmented generation (RAG) service that answers shoppers' questions about a toy and outdoor-play catalogue (848 products). Every answer is grounded in retrieved products and cites them. Questions the catalogue cannot answer are refused without calling the model.

The project started as a lab notebook (Cloud SQL + pgvector + Vertex AI + LangChain, kept in [`notebooks/`](notebooks/prototype_lab_notebook.ipynb) for reference). It was rebuilt as a service with the parts a production RAG system needs: incremental ingestion, hybrid search with filters, grounded prompting with citation checks, swappable model providers, an HTTP API, observability, an evaluation harness used as a CI quality gate, tests and containers.

> **Runs with zero credentials.** The defaults use an offline embedder and an extractive responder, so `pip install` + two commands give a working service. Switching to Vertex AI, OpenAI or a local Ollama model is a configuration change.

---

## Contents

1. [What changed from the prototype](#what-changed-from-the-prototype)
2. [Architecture](#architecture)
3. [Quick start](#quick-start)
4. [Configuration and providers](#configuration-and-providers)
5. [API](#api)
6. [Evaluation](#evaluation)
7. [Project structure](#project-structure)
8. [Testing and CI](#testing-and-ci)
9. [Deployment](#deployment)
10. [Limitations and next steps](#limitations-and-next-steps)

## What changed from the prototype

| Area | Prototype notebook | This service |
|---|---|---|
| Secrets | Project ID and DB password hard-coded in the notebook | Environment variables / `.env` (gitignored), Secret Manager on Cloud Run |
| Ingestion | Re-embeds everything every run, 7 products, batches of 3 | Incremental: content-hashed chunks, only new or changed chunks are embedded; stale rows deleted |
| Chunking | Bare description fragments with no product context | Sentence-aware chunks, each prefixed with product name, brand and category |
| Retrieval | Vector-only, fixed similarity threshold, price filter in SQL | Hybrid vector + full-text search fused with reciprocal-rank fusion, filters on price, brand, category, stock |
| Query understanding | Price range passed by hand | "under $25", "between $10 and $20" parsed from the question; explicit filters win |
| Generation | LangChain map-reduce (N+1 LLM calls per question) | One grounded call with a context budget, numbered product tags, citations validated against what was retrieved |
| Out-of-scope questions | Always answered | Evidence gate refuses before the LLM is called |
| Safety | None | Input normalisation, length limits, prompt-injection screening, hardened prompt, invalid citations stripped |
| Models | Vertex AI only, deprecated SDK | Pluggable: offline, local sentence-transformers, Vertex AI (`google-genai`), OpenAI, Ollama |
| Serving | Notebook cells | FastAPI with health/readiness probes, API-key auth, request IDs, Prometheus metrics, a small web UI |
| Quality | Eyeballed outputs | 36-question golden set, retrieval and end-to-end metrics, CI quality gate |
| Operations | None | Docker image, Docker Compose stack, GitHub Actions, runbook |

## Architecture

```mermaid
flowchart LR
    subgraph Offline["Ingestion (retail-rag ingest)"]
        CSV[Catalogue CSV] --> L[Load + clean + validate]
        L --> C[Chunk with product header]
        C --> D{Content hash changed?}
        D -- yes --> E[Embed]
        D -- no --> S
        E --> S[(Postgres + pgvector<br/>or in-memory index)]
    end
    subgraph Online["Query path (POST /v1/ask)"]
        Q[Question] --> G[Guardrails + price parsing]
        G --> R[Hybrid retrieval<br/>vector + full text + filters, RRF]
        S --> R
        R --> V{Evidence gate}
        V -- none --> X[Refuse, no LLM call]
        V -- products --> P[Grounded prompt with tags P1..Pn]
        P --> M[LLM provider]
        M --> K[Citation validation]
        K --> A[Answer + citations + timings]
    end
```

Details of each component, and why it is built that way: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Quick start

**Prerequisites:** Python 3.10 to 3.12. Docker is optional (needed only for the Postgres stack).

```bash
git clone https://github.com/craigchipendo/AI-Powered-QA-Retail-App.git
cd AI-Powered-QA-Retail-App
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[dev,vertex,openai]"
```

**1. Offline, no database, no keys** (in-memory index):

```bash
retail-rag ingest                                   # 848 products -> 2,518 chunks, ~3 s
retail-rag ask "Kiddie pool for the backyard under $25"
retail-rag serve                                    # http://127.0.0.1:8000 (UI) and /docs (OpenAPI)
```

**2. Production-like stack in Docker** (Postgres + pgvector, ingestion job, API):

```bash
cp .env.example .env
docker compose up --build                           # API on http://localhost:8000
```

**3. Real models.** Set the provider in `.env` and re-index (the embedding model is part of the index):

```bash
# Vertex AI (gcloud auth application-default login first)
RAG_EMBEDDING_PROVIDER=vertex
RAG_LLM_PROVIDER=vertex
GOOGLE_CLOUD_PROJECT=your-project-id

retail-rag ingest --rebuild
```

## Configuration and providers

All settings are environment variables (prefix `RAG_`), loaded from `.env` when present. See [.env.example](.env.example).

| Embeddings `RAG_EMBEDDING_PROVIDER` | Default model | Dim | Needs |
|---|---|---:|---|
| `hashing` (default) | feature hashing | 384 | nothing; lexical, for tests, CI and demos |
| `local` | all-MiniLM-L6-v2 | 384 | `pip install -e ".[local]"` |
| `vertex` | text-embedding-005 | 768 | GCP project + ADC |
| `openai` | text-embedding-3-small | 1536 | `OPENAI_API_KEY` |
| `ollama` | nomic-embed-text | 768 | Ollama running locally |

| LLM `RAG_LLM_PROVIDER` | Default model | Needs |
|---|---|---|
| `offline` (default) | extractive, no model | nothing |
| `vertex` | gemini-2.5-flash | GCP project + ADC |
| `openai` | gpt-4o-mini | `OPENAI_API_KEY` |
| `ollama` | llama3.2 | Ollama running locally |

Model names are defaults only; override them with `RAG_EMBEDDING_MODEL` and `RAG_LLM_MODEL`. The index records which embedding model built it, and the service refuses to start against an index built by a different one (`retail-rag ingest --rebuild` fixes it).

Key retrieval settings: `RAG_SEARCH_MODE` (hybrid, vector, keyword), `RAG_MAX_PRODUCTS` (products passed to the LLM, default 5), `RAG_CONTEXT_CHAR_BUDGET` (default 6000), `RAG_MIN_SIMILARITY` and `RAG_MIN_QUERY_COVERAGE` (the evidence gate), `RAG_HYBRID_VECTOR_WEIGHT` / `RAG_HYBRID_KEYWORD_WEIGHT`.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness |
| GET | `/ready` | Readiness: store reachable, index non-empty and built with the configured embedder |
| GET | `/metrics` | Prometheus metrics: request rate and latency, per-stage latency, answer outcomes, LLM tokens, input flags |
| POST | `/v1/search` | Retrieval only |
| POST | `/v1/ask` | Grounded answer with citations |
| POST | `/v1/products/describe` | Draft a description for a new product, styled on similar catalogue items |
| GET | `/` | Minimal web UI |

```bash
curl -s http://127.0.0.1:8000/v1/ask -H "content-type: application/json" \
  -d '{"question": "Ladder for climbing into an above ground pool", "filters": {"max_price": 120}}'
```

```json
{
  "answer": "Here are the closest matches in the catalogue for your question:\n1. **Intex Double-Sided Steel Pool Ladder for 52-Inch Above Ground Pools | 28059E** ($79.99) [P1]: ...",
  "citations": [{"ref": "P1", "product_id": "b158057daa530e7be3916c28c442c885", "product_name": "Intex Double-Sided Steel Pool Ladder for 52-Inch Above Ground Pools | 28059E", "price": 79.99, "url": "https://www.walmart.com/ip/..."}],
  "refused": false,
  "grounded": true,
  "filters_applied": {"min_price": null, "max_price": 120.0, "category": null, "brand": null, "in_stock_only": false},
  "input_flags": [],
  "model": "offline/extractive",
  "prompt_version": "answer-v1",
  "timings_ms": {"guardrails": 0.2, "retrieval": 18.0, "generation": 0.3},
  "request_id": "4f1c2a9e7b3d4c11"
}
```

Set `RAG_API_KEY` to require an `X-API-Key` header on `/v1/*`. Every response carries `X-Request-ID`, which also appears in the JSON logs.

## Evaluation

`retail-rag eval` runs a hand-labelled golden set ([eval/golden_set.jsonl](eval/golden_set.jsonl)): 29 answerable questions phrased without the product's own words, and 7 out-of-scope or adversarial ones. It reports retrieval metrics for each search mode and end-to-end metrics through the full service, and exits non-zero below the thresholds given, which is how CI uses it.

Results with the offline providers (`hashing` embedder, extractive responder):

| Store | Mode | hit@1 | hit@5 | recall@5 | MRR@10 |
|---|---|---:|---:|---:|---:|
| in-memory (BM25) | vector | 0.517 | 0.828 | 0.672 | 0.643 |
| | keyword | 0.655 | 0.966 | 0.853 | 0.782 |
| | **hybrid** | 0.621 | 0.897 | 0.744 | 0.745 |
| Postgres + pgvector | vector | 0.517 | 0.828 | 0.672 | 0.643 |
| | keyword | 0.724 | 0.862 | 0.813 | 0.803 |
| | **hybrid** | 0.655 | 0.862 | 0.759 | 0.739 |

| End to end (in-memory) | Value |
|---|---:|
| Answer cites a relevant product | 0.862 |
| Grounded (cited or refused) | 1.000 |
| False refusals on answerable questions | 0.000 |
| Out-of-scope questions refused | 1.000 |
| Latency p50 / p95 | 18 ms / 19 ms |

How to read these: the `hashing` embedder is lexical, not semantic, so keyword search beats it and hybrid sits between them. With a semantic embedder (Vertex, OpenAI, local) the vector leg improves and hybrid is expected to lead; re-run `retail-rag eval` after switching providers and tune the hybrid weights and the evidence gate on the report. Relevance labels are hand-made and may miss valid products, so retrieval numbers are lower bounds. Full reports: [reports/](reports/).

## Project structure

```
AI-Powered-QA-Retail-App/
├── src/retail_rag/
│   ├── config.py               # settings from env / .env
│   ├── schemas.py              # Pydantic contracts (Product, Chunk, AskRequest, AskResponse, ...)
│   ├── ingestion/              # loader (clean + validate), chunking, incremental pipeline
│   ├── providers/              # embeddings.py and llm.py: offline, local, vertex, openai, ollama
│   ├── store/                  # base interface, memory.py (numpy + BM25), postgres.py (pgvector + tsvector)
│   ├── retrieval.py            # hybrid retrieval, product grouping, evidence gate, context building
│   ├── generation.py           # prompt loading, generation, citation validation
│   ├── guardrails.py           # normalisation, injection screening, price parsing
│   ├── service.py              # RAGService used by the API, CLI and evaluation
│   ├── evaluation.py           # golden-set metrics and report
│   ├── metrics.py              # Prometheus metrics
│   ├── api/                    # FastAPI app + static UI
│   └── cli.py                  # `retail-rag` command
├── prompts/                    # versioned prompt templates (answer_v1, describe_v1)
├── eval/golden_set.jsonl       # labelled evaluation questions
├── data/raw/                   # product catalogue CSV
├── reports/                    # evaluation reports
├── notebooks/                  # original prototype, credentials removed
├── tests/                      # unit, API, provider and Postgres integration tests
├── docs/                       # architecture, project notes, runbook
├── Dockerfile · docker-compose.yml · .github/workflows/ci.yml
└── pyproject.toml · Makefile · .env.example
```

## Testing and CI

```bash
pytest                                          # 39 tests; Postgres tests skip unless the URL below is set
RAG_TEST_DATABASE_URL=postgresql://rag:rag@localhost:5432/rag_test pytest
```

GitHub Actions runs lint, the full test suite against a `pgvector/pgvector:pg16` service container, the evaluation quality gate on both stores (hit@5 >= 0.80, out-of-scope refusal >= 0.85) and a Docker build.

## Deployment

The container runs anywhere that runs Docker. For Google Cloud (the prototype's platform): Cloud Run for the API, Cloud SQL for PostgreSQL with `pgvector`, Vertex AI for models, Secret Manager for the database password, and a Cloud Run job for ingestion. Step-by-step commands, scaling notes, alerting rules and the re-indexing procedure are in [docs/RUNBOOK.md](docs/RUNBOOK.md).

## Limitations and next steps

* The quality numbers above use the offline providers. Vertex AI, OpenAI and local sentence-transformers paths are implemented and unit-tested against mocked clients, but were not run against live endpoints in the build environment; run `retail-rag eval` after switching.
* The Docker image and Compose stack were validated for configuration but not built in the build environment (no registry access there); CI builds the image.
* Next: a cross-encoder re-ranker, semantic caching of frequent questions, streaming responses, conversation memory, LLM-as-judge faithfulness scoring in the eval, and scheduled re-ingestion from the product feed.

---

Dataset: the retail toy dataset from Google Cloud's pgvector samples. Licensed under MIT (see [LICENSE](LICENSE)).
