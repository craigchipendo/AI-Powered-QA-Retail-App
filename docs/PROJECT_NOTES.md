# Project Notes

Findings, decisions and results from turning the lab notebook into a service. Numbers come from `reports/`.

## 1. Review of the prototype

The notebook (`notebooks/prototype_lab_notebook.ipynb`, credentials removed) loaded the catalogue into Cloud SQL, embedded descriptions with Vertex AI, searched with pgvector and summarised matches with a LangChain map-reduce chain. Issues found:

| # | Finding | Impact |
|---|---|---|
| R1 | Project ID and database password hard-coded in the notebook and committed | Credential leak; the original repository history still contains them, so the values must be treated as exposed |
| R2 | Only products 40 to 47 were embedded (`documents[40:47]`), in batches of 3 | 7 of 848 products were searchable |
| R3 | Chunks were bare description fragments (e.g. ".25-inch fitting (11070), 2 strainer grids") | Retrieved text often did not say which product it belonged to |
| R4 | Vector search only, with a hand-set similarity threshold (0.1 in one cell, 0.5 in another) | Exact terms such as brands and model numbers were matched poorly; no principled cut-off |
| R5 | Map-reduce summarisation: one LLM call per matched product plus one to combine | Cost and latency grow with the number of matches |
| R6 | Every question got an answer | Out-of-scope questions produced confident off-topic recommendations |
| R7 | The creative-description cell overwrote the retrieved matches with placeholder strings before calling the model | The generated description ignored the catalogue entirely |
| R8 | Deprecated SDK paths (`vertexai.language_models`, `langchain.chains.summarize`) | Breaks on upgrade; the notebook printed the removal date |
| R9 | No tests, no evaluation, no API, no logging | Quality and failures could not be measured |

## 2. Decisions

| Decision | Reason |
|---|---|
| Rebuild as a package with one `RAGService` used by API, CLI and eval | One code path; behaviour measured in eval is the behaviour served |
| Pluggable providers with offline defaults | Runs and tests anywhere; the cloud is a configuration choice |
| Content-hashed incremental ingestion | Re-indexing 848 products costs nothing when nothing changed (second run: 0 chunks embedded, 0.2 s on Postgres) |
| Product header on every chunk | Fixes R3 for both retrieval and generation |
| Hybrid search with RRF, filters in SQL | Fixes R4; filters are exact, ranking blends meaning and exact terms |
| Evidence gate before the LLM | Fixes R6 without spending a model call |
| Single grounded call with tagged products and citation validation | Fixes R5; every claim is traceable |
| `google-genai` SDK for Vertex AI | Replaces the deprecated SDK (R8) |
| Embedding model recorded in the index | Prevents mixing vectors from different models |

## 3. Evaluation set

36 questions in `eval/golden_set.jsonl`: 29 answerable, written in shoppers' words rather than product names (for example "Inflatable pink bird my kids can ride in the swimming pool" for flamingo floats), each with hand-labelled relevant product IDs; and 7 out-of-scope or adversarial questions (weather, sports, banking, recipes, geography, a laptop request, a prompt-injection attempt) that must be refused.

## 4. Results (offline providers)

| Store | Mode | hit@5 | MRR@10 |
|---|---|---:|---:|
| Memory | vector / keyword / hybrid | 0.828 / 0.966 / 0.897 | 0.643 / 0.782 / 0.745 |
| Postgres | vector / keyword / hybrid | 0.828 / 0.862 / 0.862 | 0.643 / 0.803 / 0.739 |

End to end (memory): relevant product cited 86%, grounded 100%, false refusals 0%, out-of-scope refused 100%, p95 latency 19 ms.

Observations:

* The vector leg is identical on both stores (same vectors, exact neighbours at this size), which confirms parity between the in-memory and pgvector paths.
* Postgres keyword search first scored hit@5 0.690 with `ts_rank_cd` over the whole chunk. Weighting the product header line (`setweight ... 'A'`) and switching to `ts_rank` raised it to 0.862.
* The hashing embedder is lexical, so keyword search leads and hybrid sits between the two. The hybrid default is kept because it is the right default for semantic embedders; weights are configurable and should be tuned on the eval after switching provider.
* The first version of the evidence gate counted any keyword match as evidence, and "What's the weather in Paris tomorrow?" was answered with pool products ("weather-resistant"). The gate now requires vector similarity above a per-model threshold, or coverage of at least 55% of the question's content words (shopping filler such as "recommend" or "best" excluded). A small plural stemmer stopped "toddlers" missing "toddler".
* Price phrases are now removed from the search text once they become filters; before that, "under $25" counted against coverage and dropped valid kiddie pools.

## 5. Not verified in the build environment

* Live calls to Vertex AI, OpenAI and Ollama (request shapes are unit-tested with fakes).
* `sentence-transformers` model download.
* `docker build` and `docker compose up` (configuration validated; CI builds the image).
* Cloud Run deployment commands in the runbook.

## 6. Next steps

1. Run the eval with Vertex AI and a local embedder, then tune `RAG_MIN_SIMILARITY` and the hybrid weights per model.
2. Add a cross-encoder re-ranker on the top 20 and measure it against hybrid.
3. Add LLM-as-judge faithfulness and answer-relevance scores to the eval for real LLM runs.
4. Semantic cache for frequent questions; streaming responses; conversation memory.
5. Schedule ingestion from the live product feed and alert on data-quality drops.
