from fastapi.testclient import TestClient
from pydantic import SecretStr

from retail_rag.api.app import create_app


def client(service, **overrides):
    s = service.s.model_copy(update=overrides)
    service.s = s
    return TestClient(create_app(s, service=service))


def test_health_ready_metrics(service):
    with client(service) as c:
        assert c.get("/health").json()["status"] == "ok"
        ready = c.get("/ready")
        assert ready.status_code == 200 and ready.json()["index"]["chunks"] > 0
        c.post("/v1/ask", json={"question": "training wheels for a bike"})
        body = c.get("/metrics").text
        assert "rag_answers_total" in body and "rag_stage_seconds" in body
        assert c.get("/").status_code == 200


def test_ask_and_search(service):
    with client(service) as c:
        r = c.post(
            "/v1/ask",
            json={"question": "snake card game", "filters": {"max_price": 20}},
            headers={"x-request-id": "abc123"},
        )
        assert r.status_code == 200 and r.headers["x-request-id"] == "abc123"
        body = r.json()
        assert body["request_id"] == "abc123" and body["citations"][0]["product_id"] == "p3"
        r = c.post("/v1/search", json={"query": "pool", "limit": 2})
        assert r.status_code == 200 and len(r.json()) <= 2


def test_validation_errors(service):
    with client(service) as c:
        assert c.post("/v1/ask", json={"question": "x"}).status_code == 422
        assert c.post("/v1/ask", json={"question": "y" * 2000}).status_code == 422
        bad = {"question": "pool", "filters": {"min_price": 50, "max_price": 10}}
        assert c.post("/v1/ask", json=bad).status_code == 422


def test_api_key_required(service):
    with client(service, api_key=SecretStr("s3cret")) as c:
        assert c.post("/v1/ask", json={"question": "pool net"}).status_code == 401
        ok = c.post("/v1/ask", json={"question": "pool net"}, headers={"x-api-key": "s3cret"})
        assert ok.status_code == 200
        assert c.get("/health").status_code == 200  # probes stay open


def test_ready_reports_empty_index(settings, tmp_path):
    from retail_rag.providers.embeddings import HashingEmbedder
    from retail_rag.providers.llm import OfflineLLM
    from retail_rag.service import RAGService
    from retail_rag.store.memory import MemoryStore

    svc = RAGService(settings, MemoryStore(tmp_path / "empty"), HashingEmbedder(64), OfflineLLM())
    with TestClient(create_app(settings, service=svc)) as c:
        assert c.get("/ready").status_code == 503
