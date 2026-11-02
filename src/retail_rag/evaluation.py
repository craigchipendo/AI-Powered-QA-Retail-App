"""Offline evaluation against a golden question set.

Retrieval (per search mode): hit@1, hit@5, recall@5, MRR@10 over answerable questions.
End to end (through RAGService.ask): answer hit rate (a cited product is relevant),
grounded rate, refusal accuracy on out-of-scope questions, false-refusal rate,
and latency percentiles.

Relevance lists are hand-labelled and may miss some valid products, so the
retrieval numbers are lower bounds. The harness exists to catch regressions:
`retail-rag eval --min-hit-at-5 0.7` exits non-zero when quality drops.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from retail_rag.schemas import AskRequest, SearchFilters
from retail_rag.service import RAGService


def load_golden(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _retrieval_metrics(service: RAGService, items: list[dict], mode: str) -> dict:
    hit1 = hit5 = recall5 = mrr = 0.0
    misses = []
    for it in items:
        rel = set(it["relevant_product_ids"])
        hits = service.retriever.retrieve(it["question"], SearchFilters(**it["filters"]), max_products=10, mode=mode)
        ranked = [h.product.product_id for h in hits]
        hit1 += bool(ranked[:1] and ranked[0] in rel)
        hit5 += bool(rel & set(ranked[:5]))
        recall5 += len(rel & set(ranked[:5])) / len(rel)
        rank = next((i for i, pid in enumerate(ranked, 1) if pid in rel), None)
        mrr += 1 / rank if rank else 0.0
        if not rel & set(ranked[:5]):
            misses.append(it["id"])
    n = max(len(items), 1)
    return {
        "hit@1": round(hit1 / n, 4),
        "hit@5": round(hit5 / n, 4),
        "recall@5": round(recall5 / n, 4),
        "mrr@10": round(mrr / n, 4),
        "misses@5": misses,
    }


def run_eval(service: RAGService, golden_path: str | Path, modes=("vector", "keyword", "hybrid")) -> dict:
    items = load_golden(golden_path)
    answerable = [i for i in items if not i["should_refuse"]]
    ood = [i for i in items if i["should_refuse"]]

    retrieval = {m: _retrieval_metrics(service, answerable, m) for m in modes}

    latencies, answer_hits, grounded, false_refusals = [], 0, 0, []
    for it in answerable:
        t0 = time.perf_counter()
        resp = service.ask(AskRequest(question=it["question"], filters=SearchFilters(**it["filters"])))
        latencies.append((time.perf_counter() - t0) * 1000)
        rel = set(it["relevant_product_ids"])
        answer_hits += any(c.product_id in rel for c in resp.citations)
        grounded += resp.grounded
        if resp.refused:
            false_refusals.append(it["id"])
    ood_refused, ood_missed = 0, []
    for it in ood:
        t0 = time.perf_counter()
        resp = service.ask(AskRequest(question=it["question"]))
        latencies.append((time.perf_counter() - t0) * 1000)
        if resp.refused:
            ood_refused += 1
        else:
            ood_missed.append(it["id"])

    lat = sorted(latencies)
    end_to_end = {
        "answer_hit_rate": round(answer_hits / max(len(answerable), 1), 4),
        "grounded_rate": round(grounded / max(len(answerable), 1), 4),
        "false_refusal_rate": round(len(false_refusals) / max(len(answerable), 1), 4),
        "false_refusals": false_refusals,
        "ood_refusal_rate": round(ood_refused / max(len(ood), 1), 4),
        "ood_not_refused": ood_missed,
        "latency_ms_p50": round(statistics.median(lat), 1) if lat else None,
        "latency_ms_p95": round(lat[int(0.95 * (len(lat) - 1))], 1) if lat else None,
    }
    return {
        "golden_set": str(golden_path),
        "questions": {"answerable": len(answerable), "out_of_scope": len(ood)},
        "embedding": service.embedder.id,
        "llm": service.llm.name,
        "store": service.store.stats().get("backend"),
        "retrieval": retrieval,
        "end_to_end": end_to_end,
    }


def to_markdown(report: dict) -> str:
    lines = [
        "# Evaluation report",
        "",
        f"Embedding `{report['embedding']}` · LLM `{report['llm']}` · store `{report['store']}` · "
        f"{report['questions']['answerable']} answerable + {report['questions']['out_of_scope']} out-of-scope questions",
        "",
        "## Retrieval",
        "",
        "| Mode | hit@1 | hit@5 | recall@5 | MRR@10 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for mode, m in report["retrieval"].items():
        lines.append(f"| {mode} | {m['hit@1']:.3f} | {m['hit@5']:.3f} | {m['recall@5']:.3f} | {m['mrr@10']:.3f} |")
    e = report["end_to_end"]
    lines += [
        "",
        "## End to end",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Answer cites a relevant product | {e['answer_hit_rate']:.3f} |",
        f"| Grounded (cited or refused) | {e['grounded_rate']:.3f} |",
        f"| False refusals on answerable questions | {e['false_refusal_rate']:.3f} |",
        f"| Out-of-scope questions refused | {e['ood_refusal_rate']:.3f} |",
        f"| Latency p50 / p95 (ms) | {e['latency_ms_p50']} / {e['latency_ms_p95']} |",
        "",
    ]
    return "\n".join(lines)
