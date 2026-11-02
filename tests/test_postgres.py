"""Integration tests against a real Postgres + pgvector (skipped unless RAG_TEST_DATABASE_URL is set)."""

import pytest

from retail_rag.ingestion.pipeline import ingest
from retail_rag.providers.embeddings import HashingEmbedder
from retail_rag.providers.llm import OfflineLLM
from retail_rag.schemas import AskRequest, SearchFilters
from retail_rag.service import RAGService
from retail_rag.store import IndexMismatchError
from tests.conftest import postgres_url

pytestmark = [pytest.mark.postgres, pytest.mark.skipif(not postgres_url(), reason="RAG_TEST_DATABASE_URL not set")]


@pytest.fixture
def pg_service(settings, catalogue_csv):
    from retail_rag.store.postgres import PostgresStore

    store, emb = PostgresStore(postgres_url()), HashingEmbedder(64)
    ingest(catalogue_csv, store, emb, rebuild=True)
    yield RAGService(settings, store, emb, OfflineLLM())
    store.close()


def test_incremental_ingest(pg_service, catalogue_csv):
    again = ingest(catalogue_csv, pg_service.store, pg_service.embedder)
    assert again.chunks_embedded == 0
    assert pg_service.store.stats()["products"] == 5


def test_hybrid_search_with_filters(pg_service):
    hits = pg_service.retriever.retrieve("net for leaves in the pool", SearchFilters())
    assert hits[0].product.product_id == "p4"
    hits = pg_service.retriever.retrieve("pool", SearchFilters(max_price=12))
    assert "p4" in {h.product.product_id for h in hits} and all(h.product.price <= 12 for h in hits)
    in_stock = pg_service.retriever.retrieve("pool", SearchFilters(in_stock_only=True, max_price=12))
    assert "p4" not in {h.product.product_id for h in in_stock}  # p4 is out of stock
    for mode in ("vector", "keyword"):
        assert pg_service.retriever.retrieve("training wheels", SearchFilters(), mode=mode)


def test_ask_end_to_end(pg_service):
    resp = pg_service.ask(AskRequest(question="snake card game"))
    assert resp.citations[0].product_id == "p3"


def test_mismatch_guard(pg_service):
    with pytest.raises(IndexMismatchError):
        pg_service.store.init("other:32", 32)
