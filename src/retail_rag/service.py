"""RAGService: the single entry point used by the API, the CLI and the evaluation harness."""

from __future__ import annotations

import time
from contextlib import contextmanager

from retail_rag import metrics
from retail_rag.config import Settings
from retail_rag.generation import REFUSAL, generate, load_prompt
from retail_rag.guardrails import merge_filters, normalize_query, parse_price_filters, screen, strip_price_phrases
from retail_rag.logging_utils import get_logger, log_event, request_id_var
from retail_rag.providers.embeddings import Embedder, build_embedder
from retail_rag.providers.llm import LLM, build_llm
from retail_rag.retrieval import Retriever
from retail_rag.schemas import (
    AskRequest,
    AskResponse,
    Citation,
    DescribeRequest,
    DescribeResponse,
    SearchFilters,
    SearchHit,
    SearchRequest,
)
from retail_rag.store import VectorStore, build_store

log = get_logger("service")


class Timer:
    def __init__(self):
        self.ms: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self.ms[name] = round(dt * 1000, 2)
            metrics.STAGE.labels(name).observe(dt)


class RAGService:
    def __init__(self, settings: Settings, store: VectorStore, embedder: Embedder, llm: LLM):
        self.s, self.store, self.embedder, self.llm = settings, store, embedder, llm
        self.retriever = Retriever(store, embedder, settings)
        self.answer_prompt = load_prompt(str(settings.prompts_dir), "answer_v1")
        self.describe_prompt = load_prompt(str(settings.prompts_dir), "describe_v1")

    @classmethod
    def from_settings(cls, settings: Settings, check_index: bool = True) -> RAGService:
        embedder = build_embedder(settings)
        store = build_store(settings)
        if check_index:
            store.check_compatible(embedder.id)
        return cls(settings, store, embedder, build_llm(settings))

    # --------------------------------------------------------------- search
    def search(self, req: SearchRequest) -> list[SearchHit]:
        query = normalize_query(req.query, self.s.max_query_chars)
        hits = self.retriever.retrieve(query, req.filters, max_products=req.limit)
        return [
            SearchHit(
                product=h.product,
                score=round(h.score, 6),
                vector_similarity=h.vector_similarity,
                matched_text=h.excerpt()[:400],
            )
            for h in hits
        ]

    # ------------------------------------------------------------------ ask
    def ask(self, req: AskRequest) -> AskResponse:
        timer = Timer()
        with timer.stage("guardrails"):
            question = normalize_query(req.question, self.s.max_query_chars)
            flags = screen(question)
            parsed = parse_price_filters(question) if req.parse_filters_from_question else None
            filters = merge_filters(req.filters, parsed) if parsed else req.filters
            search_text = strip_price_phrases(question) if parsed and not parsed.is_empty() else question
        for f in flags:
            metrics.FLAGS.labels(f).inc()

        with timer.stage("retrieval"):
            hits = self.retriever.retrieve(search_text, filters)
            evidence = [h for h in hits if self.retriever.is_evidence(h, search_text)]

        if not evidence:
            # Nothing relevant: answer without calling the model (cheaper, and it cannot hallucinate).
            metrics.OUTCOMES.labels("refused").inc()
            resp = AskResponse(
                answer=REFUSAL,
                citations=[],
                refused=True,
                grounded=True,
                filters_applied=filters,
                input_flags=flags,
                model="none",
                prompt_version=self.answer_prompt.version,
                timings_ms=timer.ms,
                request_id=request_id_var.get(),
            )
            self._log(question, resp, hits=0)
            return resp

        with timer.stage("generation"):
            out = generate(
                self.llm, self.answer_prompt, evidence, self.s.context_char_budget, "answer", question=question
            )
        outcome = "refused" if out.refused else ("answered" if out.grounded else "ungrounded")
        metrics.OUTCOMES.labels(outcome).inc()
        if out.llm.input_tokens:
            metrics.TOKENS.labels("input").inc(out.llm.input_tokens)
        if out.llm.output_tokens:
            metrics.TOKENS.labels("output").inc(out.llm.output_tokens)
        resp = AskResponse(
            answer=out.text,
            citations=out.citations,
            refused=out.refused,
            grounded=out.grounded,
            filters_applied=filters,
            input_flags=flags,
            model=out.llm.model,
            prompt_version=self.answer_prompt.version,
            timings_ms=timer.ms,
            request_id=request_id_var.get(),
        )
        self._log(question, resp, hits=len(evidence), invalid_refs=out.invalid_refs)
        return resp

    # -------------------------------------------------------------- describe
    def describe(self, req: DescribeRequest) -> DescribeResponse:
        timer = Timer()
        with timer.stage("guardrails"):
            brief = normalize_query(req.brief, self.s.max_query_chars)
        with timer.stage("retrieval"):
            hits = self.retriever.retrieve(brief, filters=SearchFilters(), max_products=2)
        with timer.stage("generation"):
            out = generate(
                self.llm, self.describe_prompt, hits, self.s.context_char_budget // 2, "describe", brief=brief
            )
        refs = [
            Citation(
                ref=h.ref,
                product_id=h.product.product_id,
                product_name=h.product.product_name,
                price=h.product.price,
                url=h.product.product_url,
            )
            for h in hits
        ]
        return DescribeResponse(
            description=out.text,
            reference_products=refs,
            model=out.llm.model,
            prompt_version=self.describe_prompt.version,
            timings_ms=timer.ms,
            request_id=request_id_var.get(),
        )

    def _log(self, question: str, resp: AskResponse, **extra):
        log_event(
            log,
            "ask",
            question_chars=len(question),
            refused=resp.refused,
            grounded=resp.grounded,
            citations=[c.product_id for c in resp.citations],
            flags=resp.input_flags,
            model=resp.model,
            timings_ms=resp.timings_ms,
            **extra,
        )

    def close(self):
        self.store.close()
