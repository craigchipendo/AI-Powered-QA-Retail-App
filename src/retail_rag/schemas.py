"""Typed data contracts shared by ingestion, retrieval, generation and the API."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class Product(BaseModel):
    product_id: str
    product_name: str
    description: str
    list_price: float | None = None
    sale_price: float | None = None
    brand: str | None = None
    category: str | None = None
    available: bool | None = None
    product_url: str | None = None

    @property
    def price(self) -> float | None:
        return self.sale_price if self.sale_price is not None else self.list_price


class Chunk(BaseModel):
    chunk_id: str
    product_id: str
    chunk_index: int
    content: str
    content_hash: str


class SearchFilters(BaseModel):
    min_price: float | None = Field(default=None, ge=0)
    max_price: float | None = Field(default=None, ge=0)
    category: str | None = Field(default=None, description="case-insensitive substring of the category path")
    brand: str | None = None
    in_stock_only: bool = False

    @field_validator("max_price")
    @classmethod
    def _range(cls, v, info):
        lo = info.data.get("min_price")
        if v is not None and lo is not None and v < lo:
            raise ValueError("max_price must be >= min_price")
        return v

    def is_empty(self) -> bool:
        return self == SearchFilters()


class RetrievedChunk(BaseModel):
    chunk: Chunk
    product: Product
    score: float  # fused rank score (higher is better)
    vector_similarity: float | None = None
    vector_rank: int | None = None
    keyword_rank: int | None = None


class Citation(BaseModel):
    ref: str  # "P1"
    product_id: str
    product_name: str
    price: float | None
    url: str | None = None


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3)
    filters: SearchFilters = Field(default_factory=SearchFilters)
    parse_filters_from_question: bool = True


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2)
    filters: SearchFilters = Field(default_factory=SearchFilters)
    limit: int = Field(default=10, ge=1, le=50)


class SearchHit(BaseModel):
    product: Product
    score: float
    vector_similarity: float | None
    matched_text: str


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    refused: bool
    grounded: bool
    filters_applied: SearchFilters
    input_flags: list[str] = Field(default_factory=list)
    model: str
    prompt_version: str
    timings_ms: dict[str, float]
    request_id: str


class DescribeRequest(BaseModel):
    brief: str = Field(..., min_length=5, description="a few words about the new product")


class DescribeResponse(BaseModel):
    description: str
    reference_products: list[Citation]
    model: str
    prompt_version: str
    timings_ms: dict[str, float]
    request_id: str
