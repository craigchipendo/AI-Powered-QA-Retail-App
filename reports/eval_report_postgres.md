# Evaluation report

Embedding `hashing:384` · LLM `offline/extractive` · store `postgres` · 29 answerable + 7 out-of-scope questions

## Retrieval

| Mode | hit@1 | hit@5 | recall@5 | MRR@10 |
| --- | ---: | ---: | ---: | ---: |
| vector | 0.517 | 0.828 | 0.672 | 0.643 |
| keyword | 0.724 | 0.862 | 0.813 | 0.803 |
| hybrid | 0.655 | 0.862 | 0.759 | 0.739 |

## End to end

| Metric | Value |
| --- | ---: |
| Answer cites a relevant product | 0.759 |
| Grounded (cited or refused) | 1.000 |
| False refusals on answerable questions | 0.000 |
| Out-of-scope questions refused | 1.000 |
| Latency p50 / p95 (ms) | 22.8 / 39.8 |
