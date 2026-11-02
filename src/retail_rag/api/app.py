"""HTTP API.

GET  /health                liveness (process is up)
GET  /ready                 readiness (store reachable, index built with the configured embedder)
GET  /metrics               Prometheus metrics
POST /v1/search             retrieval only
POST /v1/ask                grounded answer with citations
POST /v1/products/describe  product description generation
GET  /                      minimal chat UI
"""

from __future__ import annotations

import secrets
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from retail_rag import __version__, metrics
from retail_rag.config import Settings, get_settings
from retail_rag.guardrails import InputError
from retail_rag.logging_utils import configure_logging, get_logger, log_event, request_id_var
from retail_rag.schemas import (
    AskRequest,
    AskResponse,
    DescribeRequest,
    DescribeResponse,
    SearchHit,
    SearchRequest,
)
from retail_rag.service import RAGService
from retail_rag.store import IndexMismatchError

log = get_logger("api")
STATIC = Path(__file__).parent / "static"


def create_app(settings: Settings | None = None, service: RAGService | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = service or RAGService.from_settings(settings)
        log_event(
            log,
            "service started",
            embedding=app.state.service.embedder.id,
            llm=app.state.service.llm.name,
            store=settings.vector_store,
            version=__version__,
        )
        yield
        app.state.service.close()

    app = FastAPI(title="Retail RAG Assistant", version=__version__, lifespan=lifespan)
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["GET", "POST"], allow_headers=["*"]
        )

    @app.middleware("http")
    async def observe(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        token = request_id_var.set(rid)
        t0 = time.perf_counter()
        route = request.url.path
        try:
            response = await call_next(request)
        except Exception:
            metrics.REQUESTS.labels(route, "500").inc()
            log.exception("unhandled error")
            response = JSONResponse({"detail": "internal error", "request_id": rid}, status_code=500)
        dt = time.perf_counter() - t0
        metrics.LATENCY.labels(route).observe(dt)
        metrics.REQUESTS.labels(route, str(response.status_code)).inc()
        response.headers["x-request-id"] = rid
        if route.startswith("/v1"):
            log_event(
                log, "request", method=request.method, path=route, status=response.status_code, ms=round(dt * 1000, 1)
            )
        request_id_var.reset(token)
        return response

    def require_key(x_api_key: str | None = Header(default=None)):
        expected = settings.api_key.get_secret_value() if settings.api_key else None
        if expected and not (x_api_key and secrets.compare_digest(x_api_key, expected)):
            raise HTTPException(status_code=401, detail="missing or invalid X-API-Key")

    def svc(request: Request) -> RAGService:
        return request.app.state.service

    @app.exception_handler(InputError)
    async def _input_error(request: Request, exc: InputError):
        return JSONResponse({"detail": str(exc), "request_id": request_id_var.get()}, status_code=422)

    @app.get("/health")
    def health():
        return {"status": "ok", "version": __version__}

    @app.get("/ready")
    def ready(request: Request):
        s = svc(request)
        try:
            s.store.ping()
            s.store.check_compatible(s.embedder.id)
            stats = s.store.stats()
        except IndexMismatchError as exc:
            return JSONResponse({"status": "not_ready", "reason": str(exc)}, status_code=503)
        except Exception as exc:  # store unreachable
            return JSONResponse({"status": "not_ready", "reason": f"store error: {exc}"}, status_code=503)
        if not stats.get("chunks"):
            return JSONResponse(
                {"status": "not_ready", "reason": "index is empty; run `retail-rag ingest`"}, status_code=503
            )
        return {"status": "ready", "index": stats, "llm": s.llm.name}

    @app.get("/metrics")
    def prometheus():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/v1/search", response_model=list[SearchHit], dependencies=[Depends(require_key)])
    def search(req: SearchRequest, request: Request):
        return svc(request).search(req)

    @app.post("/v1/ask", response_model=AskResponse, dependencies=[Depends(require_key)])
    def ask(req: AskRequest, request: Request):
        return svc(request).ask(req)

    @app.post("/v1/products/describe", response_model=DescribeResponse, dependencies=[Depends(require_key)])
    def describe(req: DescribeRequest, request: Request):
        return svc(request).describe(req)

    @app.get("/", include_in_schema=False)
    def ui():
        return FileResponse(STATIC / "index.html")

    return app
