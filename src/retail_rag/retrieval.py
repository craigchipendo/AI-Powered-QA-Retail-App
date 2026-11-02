"""Retrieval: hybrid search, product-level grouping and context assembly."""

from __future__ import annotations

from dataclasses import dataclass, field

from retail_rag.config import Settings
from retail_rag.providers.embeddings import Embedder
from retail_rag.schemas import Product, RetrievedChunk, SearchFilters
from retail_rag.store.base import VectorStore, tokenize

# Shopping filler that says nothing about the product, ignored by the coverage gate.
QUERY_FILLER = frozenset(
    [
        "recommend",
        "recommendation",
        "good",
        "best",
        "great",
        "nice",
        "looking",
        "look",
        "want",
        "need",
        "buy",
        "find",
        "something",
        "suggest",
        "show",
        "please",
        "cheap",
        "affordable",
        "price",
        "sell",
        "selling",
        "store",
        "shop",
        "item",
        "product",
        "products",
        "get",
        "gift",
    ]
)


@dataclass
class ProductHit:
    ref: str
    product: Product
    score: float
    vector_similarity: float | None
    keyword_hit: bool
    chunks: list[RetrievedChunk] = field(default_factory=list)

    def excerpt(self) -> str:
        return "\n".join(c.chunk.content.split("\n", 1)[-1] for c in self.chunks)

    def as_context(self) -> dict:
        return {
            "ref": self.ref,
            "product_id": self.product.product_id,
            "name": self.product.product_name,
            "price": self.product.price,
            "excerpt": self.excerpt(),
        }


class Retriever:
    def __init__(self, store: VectorStore, embedder: Embedder, settings: Settings):
        self.store, self.embedder, self.s = store, embedder, settings

    def search_chunks(self, query: str, filters: SearchFilters, mode: str | None = None) -> list[RetrievedChunk]:
        qvec = self.embedder.embed_query(query)
        return self.store.search(
            query,
            qvec,
            filters,
            k=self.s.candidate_k,
            limit=self.s.candidate_k * 2,
            mode=mode or self.s.search_mode,
            w_vec=self.s.hybrid_vector_weight,
            w_kw=self.s.hybrid_keyword_weight,
            rrf_k=self.s.rrf_k,
        )

    def retrieve(
        self, query: str, filters: SearchFilters, max_products: int | None = None, mode: str | None = None
    ) -> list[ProductHit]:
        """Best-first products, each with its top chunks."""
        grouped: dict[str, ProductHit] = {}
        for rc in self.search_chunks(query, filters, mode):
            pid = rc.product.product_id
            hit = grouped.get(pid)
            if hit is None:
                hit = grouped[pid] = ProductHit("", rc.product, rc.score, rc.vector_similarity, False)
            if len(hit.chunks) < self.s.chunks_per_product:
                hit.chunks.append(rc)
            hit.keyword_hit |= rc.keyword_rank is not None
            if rc.vector_similarity is not None:
                hit.vector_similarity = max(hit.vector_similarity or -1.0, rc.vector_similarity)
        hits = sorted(grouped.values(), key=lambda h: h.score, reverse=True)[: max_products or self.s.max_products]
        for i, h in enumerate(hits, 1):
            h.ref = f"P{i}"
            h.chunks.sort(key=lambda c: c.chunk.chunk_index)
        return hits

    @property
    def min_similarity(self) -> float:
        return self.s.min_similarity if self.s.min_similarity is not None else self.embedder.default_min_similarity

    def is_evidence(self, hit: ProductHit, query: str) -> bool:
        """Relevant if semantically close enough, or if most of the query's terms appear in the product."""
        if (hit.vector_similarity or 0.0) >= self.min_similarity:
            return True
        terms = set(tokenize(query)) - QUERY_FILLER
        if not terms:
            return False
        p = hit.product
        text = set(tokenize(f"{p.product_name} {p.brand or ''} {p.category or ''} {p.description}"))
        return len(terms & text) / len(terms) >= self.s.min_query_coverage


def format_context(hits: list[ProductHit], budget: int) -> str:
    blocks, used = [], 0
    for h in hits:
        p = h.product
        price = f"${p.price:.2f}" if p.price is not None else "not listed"
        if p.sale_price is not None and p.list_price and p.sale_price < p.list_price:
            price += f" (list ${p.list_price:.2f})"
        meta = [f"Price: {price}"]
        if p.brand:
            meta.append(f"Brand: {p.brand}")
        if p.category:
            meta.append("Category: " + " > ".join(x.strip() for x in p.category.split("|")[-2:]))
        if p.available is not None:
            meta.append(f"In stock: {'yes' if p.available else 'no'}")
        block = f"[{h.ref}] {p.product_name}\n{' | '.join(meta)}\nDetails: {h.excerpt()}"
        if used + len(block) > budget:
            block = block[: max(0, budget - used)].rsplit(" ", 1)[0]
            if len(block) < 200:
                break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)
