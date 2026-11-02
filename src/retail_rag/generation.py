"""Prompt rendering, generation and citation validation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from retail_rag.providers.llm import LLM, LLMResult
from retail_rag.retrieval import ProductHit, format_context
from retail_rag.schemas import Citation

REFUSAL = "I couldn't find a product in the catalogue that matches this question."
_CITE = re.compile(r"\[(P\d+)\]")


@dataclass(frozen=True)
class Prompt:
    version: str
    system: str
    user: str


@lru_cache
def load_prompt(prompts_dir: str, name: str) -> Prompt:
    data = yaml.safe_load((Path(prompts_dir) / f"{name}.yaml").read_text(encoding="utf-8"))
    return Prompt(data["version"], data["system"].strip(), data["user"].strip())


@dataclass
class Generated:
    text: str
    citations: list[Citation]
    refused: bool
    grounded: bool
    invalid_refs: list[str]
    llm: LLMResult


def _citations(text: str, hits: list[ProductHit]) -> tuple[list[Citation], list[str]]:
    by_ref = {h.ref: h for h in hits}
    seen, cites, invalid = set(), [], []
    for ref in _CITE.findall(text):
        if ref in seen:
            continue
        seen.add(ref)
        h = by_ref.get(ref)
        if h is None:
            invalid.append(ref)
            continue
        p = h.product
        cites.append(
            Citation(ref=ref, product_id=p.product_id, product_name=p.product_name, price=p.price, url=p.product_url)
        )
    return cites, invalid


def generate(llm: LLM, prompt: Prompt, hits: list[ProductHit], budget: int, task: str, **fields) -> Generated:
    context = format_context(hits, budget)
    result = llm.generate(
        prompt.system, prompt.user.format(context=context, **fields), task=task, context=[h.as_context() for h in hits]
    )
    text = result.text.strip()
    cites, invalid = _citations(text, hits)
    for ref in invalid:  # never show the reader a citation that points nowhere
        text = text.replace(f"[{ref}]", "")
    refused = REFUSAL.lower().rstrip(".") in text.lower()
    return Generated(
        text=text, citations=cites, refused=refused, grounded=refused or bool(cites), invalid_refs=invalid, llm=result
    )
