"""Chat model providers behind one interface.

`offline` is a deterministic, extractive responder that builds an answer from the
retrieved products without any model. It keeps the service usable (and testable)
with zero credentials; real deployments set `RAG_LLM_PROVIDER` to vertex,
openai or ollama.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_exponential

from retail_rag.config import Settings

_retry = retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    retry=retry_if_not_exception_type((ValueError, TypeError, ImportError)),
)


@dataclass
class LLMResult:
    text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class LLM(ABC):
    name: str

    @abstractmethod
    def generate(
        self, system: str, user: str, *, task: str = "answer", context: list[dict] | None = None
    ) -> LLMResult: ...


class OfflineLLM(LLM):
    """Extractive fallback: no model, grounded by construction."""

    name = "offline/extractive"

    def generate(self, system, user, *, task="answer", context=None):
        context = context or []
        if not context:
            return LLMResult("I couldn't find a product in the catalogue that matches this question.", self.name)
        if task == "describe":
            ref = context[0]
            brief = re.search(r"BRIEF:\s*(.+)", user)
            brief_text = brief.group(1).strip() if brief else "this new product"
            text = (
                f"Introducing {brief_text.rstrip('.')}. In the spirit of {ref['name']} [{ref['ref']}], "
                f"it is built for everyday play: {_first_sentences(ref['excerpt'], 2)}"
            )
            return LLMResult(text, self.name)
        lines = ["Here are the closest matches in the catalogue for your question:"]
        for i, item in enumerate(context[:3], 1):
            price = f"${item['price']:.2f}" if item.get("price") is not None else "price not listed"
            lines.append(f"{i}. **{item['name']}** ({price}) [{item['ref']}]: {_first_sentences(item['excerpt'], 1)}")
        return LLMResult("\n".join(lines), self.name)


def _first_sentences(text: str, n: int, min_chars: int = 80) -> str:
    body = text.split("\n", 1)[-1]
    parts = re.split(r"(?<=[.!?])\s+", body)
    taken: list[str] = []
    for part in parts:
        taken.append(part)
        if len(taken) >= n and len(" ".join(taken)) >= min_chars:
            break
    out = " ".join(taken).strip()
    return out if len(out) <= 300 else out[:297].rsplit(" ", 1)[0] + "..."


class VertexLLM(LLM):
    def __init__(self, model: str, project: str | None, location: str, temperature: float, max_tokens: int):
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError("Install the vertex extra: pip install -e '.[vertex]'") from exc
        if not project:
            raise ValueError("GOOGLE_CLOUD_PROJECT must be set for the vertex provider")
        self.client = genai.Client(vertexai=True, project=project, location=location)
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens
        self.name = f"vertex/{model}"

    @_retry
    def generate(self, system, user, *, task="answer", context=None):
        from google.genai import types

        resp = self.client.models.generate_content(
            model=self.model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system, temperature=self.temperature, max_output_tokens=self.max_tokens
            ),
        )
        usage = getattr(resp, "usage_metadata", None)
        return LLMResult(
            resp.text or "",
            self.name,
            getattr(usage, "prompt_token_count", None),
            getattr(usage, "candidates_token_count", None),
        )


class OpenAILLM(LLM):
    def __init__(
        self, model: str, api_key: str | None, base_url: str | None, temperature: float, max_tokens: int, timeout: float
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Install the openai extra: pip install -e '.[openai]'") from exc
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set for the openai provider")
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens
        self.name = f"openai/{model}"

    @_retry
    def generate(self, system, user, *, task="answer", context=None):
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        usage = resp.usage
        return LLMResult(
            resp.choices[0].message.content or "",
            self.name,
            getattr(usage, "prompt_tokens", None),
            getattr(usage, "completion_tokens", None),
        )


class OllamaLLM(LLM):
    def __init__(self, model: str, base_url: str, temperature: float, max_tokens: int, timeout: float):
        import httpx

        self.http = httpx.Client(base_url=base_url, timeout=timeout)
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens
        self.name = f"ollama/{model}"

    @_retry
    def generate(self, system, user, *, task="answer", context=None):
        resp = self.http.post(
            "/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
            },
        )
        resp.raise_for_status()
        body = resp.json()
        return LLMResult(body["message"]["content"], self.name, body.get("prompt_eval_count"), body.get("eval_count"))


DEFAULT_MODELS = {"vertex": "gemini-2.5-flash", "openai": "gpt-4o-mini", "ollama": "llama3.2"}


def build_llm(s: Settings) -> LLM:
    provider = s.llm_provider
    model = s.llm_model or DEFAULT_MODELS.get(provider)
    if provider == "offline":
        return OfflineLLM()
    if provider == "vertex":
        return VertexLLM(model, s.gcp_project, s.gcp_location, s.llm_temperature, s.llm_max_output_tokens)
    if provider == "openai":
        key = s.openai_api_key.get_secret_value() if s.openai_api_key else None
        return OpenAILLM(model, key, s.openai_base_url, s.llm_temperature, s.llm_max_output_tokens, s.llm_timeout_s)
    if provider == "ollama":
        return OllamaLLM(model, s.ollama_base_url, s.llm_temperature, s.llm_max_output_tokens, s.llm_timeout_s)
    raise ValueError(f"Unknown LLM provider: {provider}")
