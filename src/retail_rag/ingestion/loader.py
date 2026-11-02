"""Load and clean the product catalogue."""

from __future__ import annotations

import html
import re
from pathlib import Path

import pandas as pd

from retail_rag.schemas import Product

REQUIRED = ["product_id", "product_name", "description", "list_price"]
_TAG = re.compile(r"<[^>]+>")
_SPACE = re.compile(r"\s+")
_GLUED = re.compile(r"(?<=[a-z0-9\)])\.(?=[A-Z])")  # "safe.Features" -> "safe. Features"


def clean_text(text: str) -> str:
    text = html.unescape(str(text)).replace('\\"', '"').replace("\\'", "'")
    text = _TAG.sub(" ", text)
    text = _GLUED.sub(". ", text)
    return _SPACE.sub(" ", text).strip()


def _num(value) -> float | None:
    try:
        f = float(value)
        return None if pd.isna(f) else round(f, 2)
    except (TypeError, ValueError):
        return None


def _opt(value) -> str | None:
    return None if pd.isna(value) or str(value).strip() == "" else str(value).strip()


def load_catalogue(path: str | Path, limit: int | None = None) -> tuple[list[Product], dict]:
    """Return validated products and a small data-quality report."""
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Catalogue {path} is missing required columns: {missing}")
    report = {"rows_read": len(df)}
    df = df.dropna(subset=["product_id", "product_name", "description"])
    report["dropped_missing_text"] = report["rows_read"] - len(df)
    before = len(df)
    df = df.drop_duplicates("product_id")
    report["dropped_duplicate_ids"] = before - len(df)
    if limit:
        df = df.head(limit)

    products = []
    for row in df.to_dict(orient="records"):
        desc = clean_text(row["description"])
        if len(desc) < 10:
            continue
        products.append(
            Product(
                product_id=str(row["product_id"]),
                product_name=clean_text(row["product_name"]),
                description=desc,
                list_price=_num(row.get("list_price")),
                sale_price=_num(row.get("sale_price")),
                brand=_opt(row.get("brand")),
                category=_opt(row.get("category")),
                available=bool(row["available"]) if "available" in row and not pd.isna(row["available"]) else None,
                product_url=_opt(row.get("product_url")),
            )
        )
    report["products"] = len(products)
    return products, report
