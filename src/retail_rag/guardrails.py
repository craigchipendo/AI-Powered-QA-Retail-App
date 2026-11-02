"""Input normalisation, query understanding and prompt-injection screening."""

from __future__ import annotations

import re
import unicodedata

from retail_rag.schemas import SearchFilters

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_INJECTION = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore (all |any )?(the )?(previous|prior|above) (instructions|rules)",
        r"disregard (the )?(system|previous) (prompt|instructions)",
        r"(reveal|print|show) (me )?(your|the) (system )?prompt",
        r"you are now (?!looking)",
        r"\bjailbreak\b",
        r"act as (?:an? )?(?:unfiltered|uncensored)",
    )
]
_MONEY = r"\$\s?(\d+(?:\.\d{1,2})?)|(\d+(?:\.\d{1,2})?)\s?(?:dollars|usd|bucks)\b"
_BETWEEN = re.compile(rf"between\s+(?:{_MONEY})\s+(?:and|to|-)\s+(?:{_MONEY})", re.IGNORECASE)
_RANGE = re.compile(rf"(?:{_MONEY})\s?(?:-|to)\s?(?:{_MONEY})", re.IGNORECASE)
_MAX = re.compile(
    rf"(?:under|below|less than|cheaper than|up to|at most|no more than|max(?:imum)?(?: of)?)\s+(?:{_MONEY})",
    re.IGNORECASE,
)
_MIN = re.compile(rf"(?:over|above|more than|at least|min(?:imum)?(?: of)?)\s+(?:{_MONEY})", re.IGNORECASE)


class InputError(ValueError):
    pass


def normalize_query(text: str, max_chars: int) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < 3:
        raise InputError("Question is too short.")
    if len(text) > max_chars:
        raise InputError(f"Question is longer than {max_chars} characters.")
    return text


def screen(text: str) -> list[str]:
    """Flags only: flagged questions still go through the hardened prompt, and are logged."""
    return ["possible_prompt_injection"] if any(p.search(text) for p in _INJECTION) else []


def _amounts(match: re.Match) -> list[float]:
    return [float(g) for g in match.groups() if g is not None]


def parse_price_filters(text: str) -> SearchFilters:
    """Read price constraints written in the question ("under $50", "between $20 and $40").

    Only amounts marked as money ($, dollars, usd) count, so "toys for kids under 5" is not a price.
    """
    for pattern in (_BETWEEN, _RANGE):
        m = pattern.search(text)
        if m:
            lo, hi = sorted(_amounts(m)[:2])
            return SearchFilters(min_price=lo, max_price=hi)
    lo = hi = None
    if m := _MAX.search(text):
        hi = _amounts(m)[0]
    if m := _MIN.search(text):
        lo = _amounts(m)[0]
    if lo is not None and hi is not None and hi < lo:
        lo, hi = hi, lo
    return SearchFilters(min_price=lo, max_price=hi)


def strip_price_phrases(text: str) -> str:
    """Remove the price constraints from the question so they don't act as search terms."""
    for pattern in (_BETWEEN, _RANGE, _MAX, _MIN):
        text = pattern.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip() or text


def merge_filters(explicit: SearchFilters, parsed: SearchFilters) -> SearchFilters:
    """Explicit request filters always win over values read from the question."""
    data = parsed.model_dump()
    data.update({k: v for k, v in explicit.model_dump().items() if v not in (None, False)})
    if (
        data.get("min_price") is not None
        and data.get("max_price") is not None
        and data["max_price"] < data["min_price"]
    ):
        data["min_price"] = None
    return SearchFilters(**data)
