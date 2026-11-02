# Architecture

The service has two paths that share one store: an offline ingestion path that keeps the index in step with the catalogue, and an online query path that answers questions from it. Every entry point (API, CLI, evaluation) goes through `RAGService`, so behaviour cannot drift between them.

## Components

| Component | Module | Responsibility | Why it is built this way |
|---|---|---|---|
| Settings | `config.py` | Typed settings from env / `.env` | Twelve-factor config; secrets never in code; one place to see every knob |
| Loader | `ingestion/loader.py` | Read CSV, check required columns, clean HTML/entities, drop empty or duplicate rows, report counts | Bad rows are caught and counted before they reach the index |
| Chunker | `ingestion/chunking.py` | Sentence-aware packing (600 chars, 80 overlap) with a header of product name, brand and category on every chunk | A chunk retrieved alone still says which product it describes, for both the retriever and the LLM |
| Ingestion pipeline | `ingestion/pipeline.py` | Diff chunk hashes against the index, embed only new or changed chunks, upsert, delete stale chunks and products | Re-runs are cheap and idempotent, which matters when embeddings are a paid API |
| Embedders | `providers/embeddings.py` | One interface over hashing, sentence-transformers, Vertex AI, OpenAI, Ollama; batching, retries with backoff, a query-embedding cache, document vs query task types | Providers can be swapped by configuration; transient API errors do not fail a run |
| LLMs | `providers/llm.py` | One interface over offline (extractive), Vertex AI Gemini, OpenAI, Ollama; retries, timeouts, token counts | Same reasons; the offline responder keeps tests and demos credential-free |
| Stores | `store/memory.py`, `store/postgres.py` | Persist products and chunks; filtered hybrid search | Memory for tests and small catalogues, Postgres for production, same interface and fusion logic |
| Retriever | `retrieval.py` | Run hybrid search, group chunks by product, assign tags P1..Pn, apply the evidence gate, build the context within a character budget | Answers are about products, not fragments; the gate stops irrelevant context reaching the model |
| Guardrails | `guardrails.py` | Normalise input, enforce length, screen for prompt injection, parse price phrases into filters | Cheap checks before any model call; price intent becomes an exact SQL filter instead of a fuzzy match |
| Generation | `generation.py`, `prompts/` | Render a versioned prompt, call the LLM, validate citations, strip tags that point nowhere | Every claim can be traced to a product; prompt changes are versioned and visible in responses |
| API | `api/app.py` | FastAPI routes, request IDs, API-key auth, probes, Prometheus metrics, JSON logs, a static UI | Standard operational surface for Cloud Run, Kubernetes or any container platform |
| Evaluation | `evaluation.py`, `eval/` | Golden-set retrieval and end-to-end metrics; exit code for CI | Quality changes are measured, and regressions fail the build |

## Data model (Postgres)

```sql
products(product_id PK, product_name, description, list_price, sale_price, brand, category, available,
         product_url, updated_at)
chunks(chunk_id PK = "<product_id>:<n>", product_id FK ON DELETE CASCADE, chunk_index, content,
       content_hash, embedding vector(D), tsv tsvector GENERATED, created_at)
index_metadata(key PK, value)          -- embedding_id ("vertex/text-embedding-005:768"), dim
```

Indexes: HNSW (`vector_cosine_ops`) on `chunks.embedding`, GIN on `chunks.tsv`, B-tree on effective price and `lower(brand)`. The `tsv` column weights the chunk's header line (product name, brand, category) above its body.

## Query path

1. **Guardrails.** NFKC-normalise, strip control characters, enforce 3 to 500 characters, flag injection patterns, read price constraints from the question ("under $25", "between $10 and $20"). Only amounts marked as money count, so "kids under 5" is not a price. Explicit request filters override parsed ones, and price phrases are removed from the search text.
2. **Hybrid retrieval.** One SQL statement runs a filtered HNSW vector scan and a filtered full-text search (terms OR-ed, `ts_rank` with header weighting), then fuses the two rankings with weighted reciprocal-rank fusion (`k = 60`). `hnsw.ef_search` is widened per query because pgvector applies filters after the index scan.
3. **Grouping.** Chunks are grouped into products (up to two chunks each); the top five products get tags `P1..P5`.
4. **Evidence gate.** A product counts as evidence if its vector similarity passes the embedder's threshold, or if at least 55% of the question's content words appear in its text. With no evidence the service returns the standard refusal without calling the LLM.
5. **Generation.** The versioned prompt tells the model to use only the tagged products, cite tags, copy prices exactly, refuse in a fixed sentence when nothing fits, and treat context and question as data. Context is trimmed to a character budget.
6. **Validation.** Tags the model invented are removed; an answer with no valid citation and no refusal is marked `grounded: false` and counted in metrics.

## Design decisions

| Decision | Alternatives considered | Reason |
|---|---|---|
| Thin custom pipeline, no LangChain | LangChain chains (as in the prototype) | The prototype hit two SDK deprecations; a small explicit pipeline is easier to test, trace and upgrade |
| Hybrid search with RRF | Vector only | Product names, brands and model numbers are exact-match signals that embeddings blur |
| One LLM call per question | Map-reduce over each product (prototype) | N+1 calls per question cost more and add latency with no quality gain at five products |
| Refuse before calling the LLM | Let the model decide | Cheaper, faster and cannot hallucinate; the model still refuses on its own when context is weak |
| Store the embedding model in the index | Trust configuration | Mixing vectors from two models gives silent garbage; failing loudly is safer |
| Offline providers as defaults | Require cloud credentials | Anyone can clone, test and run it; CI needs no secrets |
| psycopg 3 + pool, sync handlers | asyncpg | FastAPI runs sync handlers in a threadpool; one driver for ingestion and serving; Cloud SQL via unix socket needs no connector library |
