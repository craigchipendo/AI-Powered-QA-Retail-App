"""Incremental, idempotent ingestion.

load -> clean -> chunk -> diff against the index -> embed only new/changed chunks
-> upsert -> delete stale chunks and products.

Running it twice on the same catalogue embeds nothing the second time, which
keeps re-indexing cheap when the embedding provider is a paid API (the original
notebook re-embedded everything on each run, in batches of 3, to stay under quota).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

from retail_rag.ingestion.chunking import chunk_product
from retail_rag.ingestion.loader import load_catalogue
from retail_rag.logging_utils import get_logger, log_event
from retail_rag.providers.embeddings import Embedder
from retail_rag.store.base import VectorStore

log = get_logger("ingest")


@dataclass
class IngestReport:
    products: int
    chunks_total: int
    chunks_embedded: int
    chunks_unchanged: int
    chunks_deleted: int
    products_deleted: int
    embedding_id: str
    seconds: float
    data_quality: dict

    def as_dict(self) -> dict:
        return asdict(self)


def ingest(
    csv_path: str | Path,
    store: VectorStore,
    embedder: Embedder,
    *,
    chunk_size: int = 600,
    chunk_overlap: int = 80,
    rebuild: bool = False,
    limit: int | None = None,
) -> IngestReport:
    t0 = time.perf_counter()
    products, quality = load_catalogue(csv_path, limit=limit)
    store.init(embedder.id, embedder.dimension, rebuild=rebuild)

    chunks = [c for p in products for c in chunk_product(p, chunk_size, chunk_overlap)]
    existing = store.existing_hashes()
    todo = [c for c in chunks if existing.get(c.chunk_id) != c.content_hash]
    current_ids = {c.chunk_id for c in chunks}
    stale = [cid for cid in existing if cid not in current_ids]

    store.upsert_products(products)
    if todo:
        vectors = embedder.embed_documents([c.content for c in todo])
        store.upsert_chunks(todo, vectors)
    store.delete_chunks(stale)
    products_deleted = store.delete_products_except({p.product_id for p in products})
    store.save()

    report = IngestReport(
        products=len(products),
        chunks_total=len(chunks),
        chunks_embedded=len(todo),
        chunks_unchanged=len(chunks) - len(todo),
        chunks_deleted=len(stale),
        products_deleted=products_deleted,
        embedding_id=embedder.id,
        seconds=round(time.perf_counter() - t0, 2),
        data_quality=quality,
    )
    log_event(log, "ingest complete", **{k: v for k, v in report.as_dict().items() if k != "data_quality"})
    return report
