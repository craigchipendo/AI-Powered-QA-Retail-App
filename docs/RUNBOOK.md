# Runbook

Operating procedures for the retail RAG service. Commands run from the project root with the virtual environment active.

## 1. Local setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev,vertex,openai]"
cp .env.example .env
retail-rag ingest
pytest
```

## 2. Index operations

| Task | Command | Notes |
|---|---|---|
| Build or refresh | `retail-rag ingest` | Incremental: only new or changed chunks are embedded |
| Change embedding model | set `RAG_EMBEDDING_PROVIDER` / `RAG_EMBEDDING_MODEL`, then `retail-rag ingest --rebuild` | The service refuses to start against an index built by another model |
| Load a different catalogue | `retail-rag ingest --csv path/to/file.csv` | Needs `product_id, product_name, description, list_price`; optional `sale_price, brand, category, available, product_url` |
| Inspect | `retail-rag stats` | Product and chunk counts, embedding id |

## 3. Releasing a change

1. Change code, prompt (`prompts/*.yaml`, bump `version`) or settings.
2. `pytest` and `retail-rag eval --min-hit-at-5 0.8 --min-ood-refusal 0.85`.
3. Compare `reports/eval_report.md` with the previous run. Retrieval or refusal getting worse blocks the release.
4. Open a pull request; CI repeats the tests and the quality gate on both stores.

## 4. Switching providers

| Provider | Settings | Before first use |
|---|---|---|
| Vertex AI | `RAG_EMBEDDING_PROVIDER=vertex`, `RAG_LLM_PROVIDER=vertex`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION` | `gcloud auth application-default login`; enable the Vertex AI API |
| OpenAI | `...=openai`, `OPENAI_API_KEY` | none |
| Ollama | `...=ollama`, `OLLAMA_BASE_URL` | `ollama pull llama3.2` and `ollama pull nomic-embed-text` |
| Local embeddings | `RAG_EMBEDDING_PROVIDER=local` | `pip install -e ".[local]"` (downloads the model on first run) |

After switching the embedder, re-index and re-run the eval, then calibrate the evidence gate: raise `RAG_MIN_SIMILARITY` if out-of-scope questions get answered, lower it if answerable ones get refused.

## 5. Deploying to Google Cloud

```bash
PROJECT=your-project REGION=us-central1 INSTANCE=retail-ins
gcloud services enable run.googleapis.com sqladmin.googleapis.com aiplatform.googleapis.com secretmanager.googleapis.com

# Database (Postgres 16 has pgvector available as an extension)
gcloud sql instances create $INSTANCE --database-version=POSTGRES_16 --region=$REGION --tier=db-custom-1-3840
gcloud sql databases create retail --instance=$INSTANCE
gcloud sql users create rag --instance=$INSTANCE --password="$(openssl rand -base64 24)"

# Store the connection string in Secret Manager (never in the image or the repo)
printf 'postgresql://rag:PASSWORD@/retail?host=/cloudsql/%s:%s:%s' $PROJECT $REGION $INSTANCE | \
  gcloud secrets create rag-database-url --data-file=-

# Build and deploy the API
gcloud builds submit --tag $REGION-docker.pkg.dev/$PROJECT/apps/retail-rag:1.0.0
gcloud run deploy retail-rag --image $REGION-docker.pkg.dev/$PROJECT/apps/retail-rag:1.0.0 --region $REGION \
  --add-cloudsql-instances $PROJECT:$REGION:$INSTANCE \
  --set-secrets RAG_DATABASE_URL=rag-database-url:latest,RAG_API_KEY=rag-api-key:latest \
  --set-env-vars RAG_VECTOR_STORE=postgres,RAG_EMBEDDING_PROVIDER=vertex,RAG_LLM_PROVIDER=vertex,GOOGLE_CLOUD_PROJECT=$PROJECT \
  --min-instances 1 --concurrency 20 --cpu 1 --memory 1Gi

# Ingestion as a Cloud Run job (run on deploy and on a schedule)
gcloud run jobs create retail-rag-ingest --image $REGION-docker.pkg.dev/$PROJECT/apps/retail-rag:1.0.0 \
  --region $REGION --set-cloudsql-instances $PROJECT:$REGION:$INSTANCE --command retail-rag --args ingest \
  --set-secrets RAG_DATABASE_URL=rag-database-url:latest \
  --set-env-vars RAG_VECTOR_STORE=postgres,RAG_EMBEDDING_PROVIDER=vertex,GOOGLE_CLOUD_PROJECT=$PROJECT
gcloud run jobs execute retail-rag-ingest --region $REGION
```

The Cloud Run service account needs `roles/cloudsql.client`, `roles/aiplatform.user` and `roles/secretmanager.secretAccessor`. Point Cloud Run's health check at `/ready` so instances only receive traffic once the index is usable.

## 6. Monitoring and alerts

Scrape `/metrics` (Managed Prometheus, Grafana Agent, or any Prometheus). The image runs one uvicorn worker per container so each scrape sees the whole process; scale with more instances rather than more workers.

| Alert | Expression (PromQL sketch) | Likely cause | Action |
|---|---|---|---|
| Error rate | `sum(rate(rag_http_requests_total{status=~"5.."}[5m])) / sum(rate(rag_http_requests_total[5m])) > 0.02` | Store or provider outage | Check logs by `request_id`; check provider status |
| Slow answers | `histogram_quantile(0.95, sum by (le) (rate(rag_http_request_seconds_bucket{route="/v1/ask"}[5m]))) > 5` | LLM latency, cold instances | Check `rag_stage_seconds` by stage; raise min instances |
| Refusal spike | `rate(rag_answers_total{outcome="refused"}[1h]) / rate(rag_answers_total[1h]) > 0.4` | Catalogue gap, new demand, broken index | Read the refused questions in logs; re-ingest; extend the catalogue |
| Ungrounded answers | `rate(rag_answers_total{outcome="ungrounded"}[1h]) > 0` | Prompt or model change | Roll back the prompt version; re-run the eval |
| Injection attempts | `increase(rag_input_flags_total{flag="possible_prompt_injection"}[1h]) > 20` | Abuse | Rate-limit at the gateway; review flagged requests |
| Not ready | `/ready` returns 503 | Empty index, model mismatch, DB down | The response body says which |

Logs are JSON, one line per event, each carrying `request_id`; `/v1/ask` logs refusal, grounding, cited product IDs, flags, model and per-stage timings, and never the full question text.

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `IndexMismatchError` at startup | The embedding model changed: `retail-rag ingest --rebuild` |
| `/ready` says the index is empty | Run `retail-rag ingest` against the same store the API uses |
| Every answer is a refusal | The evidence gate is too strict for this embedder: lower `RAG_MIN_SIMILARITY` and check with `retail-rag eval` |
| Off-topic questions get answered | Raise `RAG_MIN_SIMILARITY` or `RAG_MIN_QUERY_COVERAGE` |
| `OPENAI_API_KEY must be set` / `GOOGLE_CLOUD_PROJECT must be set` | Add it to `.env` or the environment |
| Windows install fails with a long path error | Enable long paths, or create the venv in a short folder such as `C:\venvs\rag` |
