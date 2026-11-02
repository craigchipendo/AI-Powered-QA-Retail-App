"""Sentence-aware chunking with a contextual header on every chunk.

Each chunk starts with the product's name, brand and category, so a chunk
retrieved on its own still says which product it describes (the original
notebook embedded bare description fragments, so a fragment such as
".25-inch fitting, 2 strainer grids" carried no product context at all).
"""

from __future__ import annotations

import hashlib
import re

from retail_rag.schemas import Chunk, Product

_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.split(text) if s.strip()]


def _hard_wrap(sentence: str, size: int) -> list[str]:
    words, out, cur = sentence.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > size:
            out.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        out.append(cur)
    return out


def pack(sentences: list[str], size: int, overlap: int) -> list[str]:
    """Greedily pack sentences into chunks of <= size chars, carrying ~overlap chars forward."""
    units: list[str] = []
    for s in sentences:
        units.extend(_hard_wrap(s, size) if len(s) > size else [s])
    chunks, cur = [], []
    for unit in units:
        if cur and len(" ".join(cur)) + 1 + len(unit) > size:
            chunks.append(" ".join(cur))
            carry, total = [], 0
            for prev in reversed(cur):
                if total + len(prev) > overlap:
                    break
                carry.insert(0, prev)
                total += len(prev) + 1
            cur = carry
        cur.append(unit)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def chunk_header(p: Product) -> str:
    parts = [f"Product: {p.product_name}"]
    if p.brand:
        parts.append(f"Brand: {p.brand}")
    if p.category:
        parts.append("Category: " + " > ".join(c.strip() for c in p.category.split("|")[-2:]))
    return " | ".join(parts)


def chunk_product(p: Product, size: int = 600, overlap: int = 80) -> list[Chunk]:
    header = chunk_header(p)
    bodies = pack(split_sentences(p.description), size, overlap) or [p.description[:size]]
    chunks = []
    for i, body in enumerate(bodies):
        content = f"{header}\n{body}"
        chunks.append(
            Chunk(
                chunk_id=f"{p.product_id}:{i}",
                product_id=p.product_id,
                chunk_index=i,
                content=content,
                content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest()[:16],
            )
        )
    return chunks
