"""In-memory store (numpy cosine + BM25), persisted to disk between runs.

Good for tests, CI, demos and small catalogues. Same interface and fusion logic
as the Postgres store.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from retail_rag.schemas import Chunk, Product, RetrievedChunk, SearchFilters
from retail_rag.store.base import VectorStore, rrf, tokenize


def matches(p: Product, f: SearchFilters) -> bool:
    price = p.price
    if f.min_price is not None and (price is None or price < f.min_price):
        return False
    if f.max_price is not None and (price is None or price > f.max_price):
        return False
    if f.category and f.category.lower() not in (p.category or "").lower():
        return False
    if f.brand and f.brand.lower() != (p.brand or "").lower():
        return False
    return not (f.in_stock_only and p.available is False)


class MemoryStore(VectorStore):
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.products: dict[str, Product] = {}
        self.chunks: dict[str, Chunk] = {}
        self.vectors: dict[str, np.ndarray] = {}
        self.meta: dict = {}
        self._bm25 = None
        if self.path and (self.path / "meta.json").exists():
            self._load()

    # ------------------------------------------------------------ lifecycle
    def init(self, embedding_id, dim, rebuild=False):
        if rebuild:
            self.products, self.chunks, self.vectors = {}, {}, {}
        elif self.meta.get("embedding_id") not in (None, embedding_id):
            self.check_compatible(embedding_id)
        self.meta = {"embedding_id": embedding_id, "dim": dim}

    def save(self):
        if not self.path:
            return
        self.path.mkdir(parents=True, exist_ok=True)
        ids = list(self.chunks)
        (self.path / "meta.json").write_text(json.dumps(self.meta))
        (self.path / "products.json").write_text(json.dumps([p.model_dump() for p in self.products.values()]))
        (self.path / "chunks.json").write_text(json.dumps([self.chunks[i].model_dump() for i in ids]))
        np.save(self.path / "vectors.npy", np.stack([self.vectors[i] for i in ids]) if ids else np.zeros((0, 1)))

    def _load(self):
        self.meta = json.loads((self.path / "meta.json").read_text())
        self.products = {p["product_id"]: Product(**p) for p in json.loads((self.path / "products.json").read_text())}
        chunks = [Chunk(**c) for c in json.loads((self.path / "chunks.json").read_text())]
        vecs = np.load(self.path / "vectors.npy")
        self.chunks = {c.chunk_id: c for c in chunks}
        self.vectors = {c.chunk_id: vecs[i] for i, c in enumerate(chunks)}

    # -------------------------------------------------------------- writes
    def upsert_products(self, products):
        self.products.update({p.product_id: p for p in products})

    def existing_hashes(self):
        return {cid: c.content_hash for cid, c in self.chunks.items()}

    def upsert_chunks(self, chunks, vectors):
        for c, v in zip(chunks, vectors, strict=True):
            self.chunks[c.chunk_id] = c
            self.vectors[c.chunk_id] = np.asarray(v, dtype="float32")
        self._bm25 = None

    def delete_chunks(self, chunk_ids):
        for cid in chunk_ids:
            self.chunks.pop(cid, None)
            self.vectors.pop(cid, None)
        self._bm25 = None

    def delete_products_except(self, keep_ids):
        gone = [pid for pid in self.products if pid not in keep_ids]
        for pid in gone:
            del self.products[pid]
        self.delete_chunks([cid for cid, c in self.chunks.items() if c.product_id in gone])
        return len(gone)

    # -------------------------------------------------------------- search
    def _build_bm25(self):
        docs = {cid: tokenize(c.content) for cid, c in self.chunks.items()}
        df = Counter(t for toks in docs.values() for t in set(toks))
        n = max(len(docs), 1)
        avg = sum(len(t) for t in docs.values()) / n
        idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self._bm25 = (docs, idf, avg)

    def _bm25_scores(self, query, candidates, k1=1.5, b=0.75):
        if self._bm25 is None:
            self._build_bm25()
        docs, idf, avg = self._bm25
        q = set(tokenize(query))
        scores = {}
        for cid in candidates:
            toks = docs.get(cid, [])
            tf = Counter(toks)
            s = sum(
                idf.get(t, 0) * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(toks) / avg)) for t in q if t in tf
            )
            if s > 0:
                scores[cid] = s
        return scores

    def search(self, query_text, query_vec, filters, *, k, limit, mode="hybrid", w_vec=1.0, w_kw=1.0, rrf_k=60):
        cands = [cid for cid, c in self.chunks.items() if matches(self.products[c.product_id], filters)]
        if not cands:
            return []
        vec_rank, sims = {}, {}
        if mode in ("hybrid", "vector"):
            mat = np.stack([self.vectors[c] for c in cands])
            sim = mat @ np.asarray(query_vec, dtype="float32")
            order = np.argsort(-sim)[:k]
            for r, i in enumerate(order, 1):
                vec_rank[cands[i]] = r
            sims = {cands[i]: float(sim[i]) for i in range(len(cands))}
        kw_rank = {}
        if mode in ("hybrid", "keyword"):
            bm = self._bm25_scores(query_text, cands)
            for r, cid in enumerate(sorted(bm, key=bm.get, reverse=True)[:k], 1):
                kw_rank[cid] = r
        weights = {"vec": w_vec, "kw": w_kw}
        fused = {
            cid: rrf({"vec": vec_rank.get(cid), "kw": kw_rank.get(cid)}, weights, rrf_k)
            for cid in set(vec_rank) | set(kw_rank)
        }
        out = []
        for cid in sorted(fused, key=fused.get, reverse=True)[:limit]:
            c = self.chunks[cid]
            out.append(
                RetrievedChunk(
                    chunk=c,
                    product=self.products[c.product_id],
                    score=fused[cid],
                    vector_similarity=sims.get(cid),
                    vector_rank=vec_rank.get(cid),
                    keyword_rank=kw_rank.get(cid),
                )
            )
        return out

    def stats(self):
        return {"backend": "memory", "products": len(self.products), "chunks": len(self.chunks), **self.meta}

    def metadata(self):
        return dict(self.meta)
