"""Embedding providers behind one interface.

| provider | model (default)             | dim  | needs                          |
|----------|-----------------------------|------|--------------------------------|
| hashing  | feature hashing, offline    | 384  | nothing (tests, CI, demo)      |
| local    | all-MiniLM-L6-v2            | 384  | `pip install .[local]`         |
| vertex   | text-embedding-005          | 768  | GCP project + ADC credentials  |
| openai   | text-embedding-3-small      | 1536 | OPENAI_API_KEY                 |
| ollama   | nomic-embed-text            | 768  | a running Ollama server        |

Documents and queries are embedded through separate methods so providers that
support asymmetric retrieval (Vertex task types) can use it.
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from functools import lru_cache

import numpy as np
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_exponential

from retail_rag.config import Settings
from retail_rag.logging_utils import get_logger

log = get_logger("embeddings")

_retry = retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    retry=retry_if_not_exception_type((ValueError, TypeError, ImportError)),
)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


class Embedder(ABC):
    name: str
    dimension: int
    # Cosine similarity above which a hit is treated as relevant evidence. Starting points
    # only: each model has its own similarity scale, so tune it on the eval set.
    default_min_similarity: float = 0.5

    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size

    @abstractmethod
    def _embed(self, texts: list[str], kind: str) -> list[list[float]]: ...

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            out.extend(self._embed(texts[i : i + self.batch_size], "document"))
        return _normalize(np.asarray(out, dtype="float32"))

    def embed_query(self, text: str) -> np.ndarray:
        return _cached_query(self, text)

    @property
    def id(self) -> str:
        return f"{self.name}:{self.dimension}"


@lru_cache(maxsize=2048)
def _cached_query(embedder: Embedder, text: str) -> np.ndarray:
    vec = _normalize(np.asarray(embedder._embed([text], "query"), dtype="float32"))[0]
    vec.setflags(write=False)
    return vec


# --------------------------------------------------------------- offline
_TOKEN = re.compile(r"[a-z0-9]+")


class HashingEmbedder(Embedder):
    """Deterministic, dependency-free lexical embedding (word uni/bi-grams + char trigrams).

    Not semantic. It exists so the whole pipeline (and CI) runs with no model
    download and no API key; swap in a real provider for quality.
    """

    def __init__(self, dim: int = 384, batch_size: int = 256):
        super().__init__(batch_size)
        self.dimension = dim
        self.name = "hashing"
        self.default_min_similarity = 1.01  # lexical hashing similarities are not semantic: rely on coverage

    def _features(self, text: str):
        words = _TOKEN.findall(text.lower())
        feats = [f"w:{w}" for w in words]
        feats += [f"b:{a}_{b}" for a, b in zip(words, words[1:], strict=False)]
        for w in words:
            padded = f"#{w}#"
            feats += [f"c:{padded[i : i + 3]}" for i in range(len(padded) - 2)]
        return feats

    def _embed(self, texts, kind):
        out = np.zeros((len(texts), self.dimension), dtype="float32")
        for row, text in enumerate(texts):
            for f in self._features(text):
                h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little")
                weight = 0.5 if f.startswith("c:") else 1.0
                out[row, h % self.dimension] += weight if (h >> 63) & 1 else -weight
        return out.tolist()


# ------------------------------------------------------------- providers
class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 64):
        super().__init__(batch_size)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("Install the local extra: pip install -e '.[local]'") from exc
        self.model = SentenceTransformer(model)
        self.dimension = int(self.model.get_sentence_embedding_dimension())
        self.name = f"local/{model.split('/')[-1]}"
        self.default_min_similarity = 0.45

    def _embed(self, texts, kind):
        return self.model.encode(texts, normalize_embeddings=True).tolist()


class VertexEmbedder(Embedder):
    DIMS = {"text-embedding-005": 768, "text-embedding-004": 768, "text-multilingual-embedding-002": 768}

    def __init__(self, model: str, project: str | None, location: str, batch_size: int = 32):
        super().__init__(min(batch_size, 250))
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError("Install the vertex extra: pip install -e '.[vertex]'") from exc
        if not project:
            raise ValueError("GOOGLE_CLOUD_PROJECT must be set for the vertex provider")
        self.client = genai.Client(vertexai=True, project=project, location=location)
        self.model = model
        self.dimension = self.DIMS.get(model) or len(self._embed(["dimension probe"], "query")[0])
        self.name = f"vertex/{model}"
        self.default_min_similarity = 0.6

    @_retry
    def _embed(self, texts, kind):
        from google.genai import types

        task = "RETRIEVAL_DOCUMENT" if kind == "document" else "RETRIEVAL_QUERY"
        resp = self.client.models.embed_content(
            model=self.model, contents=texts, config=types.EmbedContentConfig(task_type=task)
        )
        return [e.values for e in resp.embeddings]


class OpenAIEmbedder(Embedder):
    DIMS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}

    def __init__(self, model: str, api_key: str | None, base_url: str | None, batch_size: int = 128):
        super().__init__(batch_size)
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Install the openai extra: pip install -e '.[openai]'") from exc
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set for the openai provider")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.dimension = self.DIMS.get(model) or len(self._embed(["dimension probe"], "query")[0])
        self.name = f"openai/{model}"
        self.default_min_similarity = 0.45

    @_retry
    def _embed(self, texts, kind):
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]


class OllamaEmbedder(Embedder):
    def __init__(self, model: str, base_url: str, batch_size: int = 32, timeout: float = 60.0):
        super().__init__(batch_size)
        import httpx

        self.http = httpx.Client(base_url=base_url, timeout=timeout)
        self.model = model
        self.dimension = len(self._embed(["dimension probe"], "query")[0])
        self.name = f"ollama/{model}"
        self.default_min_similarity = 0.6

    @_retry
    def _embed(self, texts, kind):
        resp = self.http.post("/api/embed", json={"model": self.model, "input": texts})
        resp.raise_for_status()
        return resp.json()["embeddings"]


DEFAULT_MODELS = {
    "local": "sentence-transformers/all-MiniLM-L6-v2",
    "vertex": "text-embedding-005",
    "openai": "text-embedding-3-small",
    "ollama": "nomic-embed-text",
}


def build_embedder(s: Settings) -> Embedder:
    provider = s.embedding_provider
    model = s.embedding_model or DEFAULT_MODELS.get(provider)
    if provider == "hashing":
        return HashingEmbedder(dim=s.embedding_dim)
    if provider == "local":
        return SentenceTransformerEmbedder(model, s.embedding_batch_size)
    if provider == "vertex":
        return VertexEmbedder(model, s.gcp_project, s.gcp_location, s.embedding_batch_size)
    if provider == "openai":
        key = s.openai_api_key.get_secret_value() if s.openai_api_key else None
        return OpenAIEmbedder(model, key, s.openai_base_url, s.embedding_batch_size)
    if provider == "ollama":
        return OllamaEmbedder(model, s.ollama_base_url, s.embedding_batch_size)
    raise ValueError(f"Unknown embedding provider: {provider}")
