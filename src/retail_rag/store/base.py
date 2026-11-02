"""Vector store interface shared by the in-memory and Postgres/pgvector backends."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

import numpy as np

from retail_rag.schemas import Chunk, Product, RetrievedChunk, SearchFilters

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "of",
        "at",
        "by",
        "for",
        "with",
        "about",
        "to",
        "from",
        "in",
        "on",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "you",
        "your",
        "we",
        "our",
        "i",
        "me",
        "my",
        "they",
        "them",
        "their",
        "what",
        "which",
        "who",
        "how",
        "can",
        "could",
        "would",
        "should",
        "will",
        "any",
        "some",
        "there",
        "here",
        "as",
        "so",
        "than",
        "too",
        "very",
        "just",
    ]
)
_TOKEN = re.compile(r"[a-z0-9]+")


def stem(word: str) -> str:
    """Tiny plural stemmer (toddlers -> toddler, batteries -> battery, boxes -> box).

    Postgres full-text search uses the Snowball English stemmer; this keeps the
    in-memory BM25 and the evidence gate roughly in line with it without a dependency.
    """
    if len(word) <= 3 or word.endswith("ss"):
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("ches", "shes", "xes", "sses")):
        return word[:-2]
    if word.endswith("s"):
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


class IndexMismatchError(RuntimeError):
    """The stored vectors were built with a different embedding model than the one configured."""


class VectorStore(ABC):
    @abstractmethod
    def init(self, embedding_id: str, dim: int, rebuild: bool = False) -> None: ...

    @abstractmethod
    def upsert_products(self, products: list[Product]) -> None: ...

    @abstractmethod
    def existing_hashes(self) -> dict[str, str]: ...

    @abstractmethod
    def upsert_chunks(self, chunks: list[Chunk], vectors: np.ndarray) -> None: ...

    @abstractmethod
    def delete_chunks(self, chunk_ids: list[str]) -> None: ...

    @abstractmethod
    def delete_products_except(self, keep_ids: set[str]) -> int: ...

    @abstractmethod
    def search(
        self,
        query_text: str,
        query_vec: np.ndarray,
        filters: SearchFilters,
        *,
        k: int,
        limit: int,
        mode: str = "hybrid",
        w_vec: float = 1.0,
        w_kw: float = 1.0,
        rrf_k: int = 60,
    ) -> list[RetrievedChunk]: ...

    @abstractmethod
    def stats(self) -> dict: ...

    @abstractmethod
    def metadata(self) -> dict: ...

    def check_compatible(self, embedding_id: str) -> None:
        stored = self.metadata().get("embedding_id")
        if stored and stored != embedding_id:
            raise IndexMismatchError(
                f"Index was built with '{stored}' but the service is configured for '{embedding_id}'. "
                "Re-index with `retail-rag ingest --rebuild`."
            )

    def ping(self) -> bool:
        return True

    def save(self) -> None:  # persistence hook (memory store)
        return None

    def close(self) -> None:
        return None


def rrf(ranks: dict[str, int | None], weights: dict[str, float], k: int) -> float:
    """Weighted reciprocal-rank fusion."""
    return sum(weights[name] / (k + r) for name, r in ranks.items() if r is not None)
