"""`retail-rag` command line.

retail-rag ingest [--rebuild]       build or update the index
retail-rag search "query"           retrieval only
retail-rag ask "question"           full RAG answer
retail-rag describe "brief"         generate a product description
retail-rag serve                    run the API on :8000
retail-rag eval                     golden-set evaluation (CI quality gate)
retail-rag stats                    index statistics
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from retail_rag.config import PROJECT_ROOT, get_settings
from retail_rag.logging_utils import configure_logging


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def _service(check_index: bool = True):
    from retail_rag.service import RAGService

    return RAGService.from_settings(get_settings(), check_index=check_index)


def cmd_ingest(args):
    from retail_rag.ingestion.pipeline import ingest
    from retail_rag.providers.embeddings import build_embedder
    from retail_rag.store import build_store

    s = get_settings()
    store, embedder = build_store(s), build_embedder(s)
    try:
        report = ingest(
            args.csv or s.catalogue_csv,
            store,
            embedder,
            chunk_size=s.chunk_size,
            chunk_overlap=s.chunk_overlap,
            rebuild=args.rebuild,
            limit=args.limit,
        )
        _print(report.as_dict())
    finally:
        store.close()


def _filters(args):
    from retail_rag.schemas import SearchFilters

    return SearchFilters(
        min_price=args.min_price,
        max_price=args.max_price,
        category=args.category,
        brand=args.brand,
        in_stock_only=args.in_stock,
    )


def cmd_search(args):
    from retail_rag.schemas import SearchRequest

    svc = _service()
    try:
        hits = svc.search(SearchRequest(query=args.query, filters=_filters(args), limit=args.limit))
        for h in hits:
            price = f"${h.product.price:.2f}" if h.product.price is not None else "-"
            print(f"{h.score:.4f}  {price:>9}  {h.product.product_name[:90]}")
    finally:
        svc.close()


def cmd_ask(args):
    from retail_rag.schemas import AskRequest

    svc = _service()
    try:
        resp = svc.ask(AskRequest(question=args.question, filters=_filters(args)))
        if args.json:
            _print(resp.model_dump())
        else:
            print(resp.answer)
            print("\nSources:")
            for c in resp.citations:
                print(f"  [{c.ref}] {c.product_name} ({c.product_id})")
            print(f"\nmodel={resp.model} prompt={resp.prompt_version} timings_ms={resp.timings_ms}")
    finally:
        svc.close()


def cmd_describe(args):
    from retail_rag.schemas import DescribeRequest

    svc = _service()
    try:
        resp = svc.describe(DescribeRequest(brief=args.brief))
        print(resp.description)
        print("\nStyle references: " + ", ".join(f"[{c.ref}] {c.product_name}" for c in resp.reference_products))
    finally:
        svc.close()


def cmd_serve(args):
    import uvicorn

    uvicorn.run(
        "retail_rag.api.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="warning",
    )


def cmd_eval(args):
    from retail_rag.evaluation import run_eval, to_markdown

    svc = _service()
    try:
        report = run_eval(svc, args.golden)
    finally:
        svc.close()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval_report.json").write_text(json.dumps(report, indent=2))
    (out / "eval_report.md").write_text(to_markdown(report))
    print(to_markdown(report))
    hybrid = report["retrieval"].get("hybrid", {})
    failures = []
    if args.min_hit_at_5 is not None and hybrid.get("hit@5", 0) < args.min_hit_at_5:
        failures.append(f"hybrid hit@5 {hybrid.get('hit@5')} < {args.min_hit_at_5}")
    if args.min_ood_refusal is not None and report["end_to_end"]["ood_refusal_rate"] < args.min_ood_refusal:
        failures.append(f"ood refusal {report['end_to_end']['ood_refusal_rate']} < {args.min_ood_refusal}")
    if failures:
        print("QUALITY GATE FAILED: " + "; ".join(failures), file=sys.stderr)
        return 1
    return 0


def cmd_stats(args):
    from retail_rag.store import build_store

    store = build_store(get_settings())
    try:
        _print(store.stats())
    finally:
        store.close()


def _add_filter_args(p):
    p.add_argument("--min-price", type=float)
    p.add_argument("--max-price", type=float)
    p.add_argument("--category")
    p.add_argument("--brand")
    p.add_argument("--in-stock", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="retail-rag", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("ingest", help="build or incrementally update the index")
    s.add_argument("--csv", type=Path)
    s.add_argument("--rebuild", action="store_true", help="drop and rebuild (needed after changing embedding model)")
    s.add_argument("--limit", type=int)
    s.set_defaults(fn=cmd_ingest)

    s = sub.add_parser("search", help="retrieval only")
    s.add_argument("query")
    s.add_argument("--limit", type=int, default=10)
    _add_filter_args(s)
    s.set_defaults(fn=cmd_search)

    s = sub.add_parser("ask", help="answer a question with citations")
    s.add_argument("question")
    s.add_argument("--json", action="store_true")
    _add_filter_args(s)
    s.set_defaults(fn=cmd_ask)

    s = sub.add_parser("describe", help="write a description for a new product")
    s.add_argument("brief")
    s.set_defaults(fn=cmd_describe)

    s = sub.add_parser("serve", help="run the HTTP API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--workers", type=int, default=1)
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("eval", help="evaluate on the golden set")
    s.add_argument("--golden", type=Path, default=PROJECT_ROOT / "eval" / "golden_set.jsonl")
    s.add_argument("--out", default=str(PROJECT_ROOT / "reports"))
    s.add_argument("--min-hit-at-5", type=float)
    s.add_argument("--min-ood-refusal", type=float)
    s.set_defaults(fn=cmd_eval)

    sub.add_parser("stats", help="index statistics").set_defaults(fn=cmd_stats)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    s = get_settings()
    configure_logging(s.log_level if args.command == "serve" else "WARNING", json_logs=s.log_json)
    return args.fn(args) or 0


if __name__ == "__main__":
    sys.exit(main())
