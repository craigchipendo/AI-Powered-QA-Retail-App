import numpy as np
import pytest

from retail_rag.config import Settings
from retail_rag.providers import embeddings, llm
from retail_rag.providers.embeddings import HashingEmbedder, build_embedder
from retail_rag.providers.llm import OfflineLLM, build_llm


def test_hashing_embedder_normalised_and_deterministic():
    e = HashingEmbedder(128)
    a = e.embed_documents(["kiddie pool", "kiddie pool", "card game"])
    assert a.shape == (3, 128)
    assert np.allclose(np.linalg.norm(a, axis=1), 1)
    assert np.allclose(a[0], a[1]) and a[0] @ a[2] < a[0] @ a[1]
    assert np.allclose(e.embed_query("kiddie pool"), a[0])


def test_defaults_need_no_credentials():
    s = Settings()
    assert isinstance(build_embedder(s), HashingEmbedder) and isinstance(build_llm(s), OfflineLLM)


def test_missing_credentials_fail_fast():
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        build_llm(Settings(llm_provider="openai", OPENAI_API_KEY=None))
    with pytest.raises(ValueError, match="GOOGLE_CLOUD_PROJECT"):
        build_embedder(Settings(embedding_provider="vertex", GOOGLE_CLOUD_PROJECT=None))


class _Resp:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class _FakeHttp:
    def __init__(self, payload):
        self.payload, self.calls = payload, []

    def post(self, path, json):
        self.calls.append((path, json))
        return _Resp(self.payload(json) if callable(self.payload) else self.payload)


def test_ollama_request_shapes(monkeypatch):
    emb = embeddings.OllamaEmbedder.__new__(embeddings.OllamaEmbedder)
    emb.batch_size, emb.model = 8, "nomic-embed-text"
    emb.http = _FakeHttp(lambda body: {"embeddings": [[1.0, 0.0, 0.0] for _ in body["input"]]})
    vecs = emb.embed_documents(["a", "b"])
    assert vecs.shape == (2, 3) and emb.http.calls[0][0] == "/api/embed"

    chat = llm.OllamaLLM.__new__(llm.OllamaLLM)
    chat.model, chat.temperature, chat.max_tokens, chat.name = "llama3.2", 0.2, 64, "ollama/llama3.2"
    chat.http = _FakeHttp({"message": {"content": "Hi [P1]"}, "prompt_eval_count": 10, "eval_count": 3})
    out = chat.generate("sys", "user")
    path, body = chat.http.calls[0]
    assert path == "/api/chat" and body["stream"] is False and body["messages"][0]["role"] == "system"
    assert out.text == "Hi [P1]" and out.input_tokens == 10


def test_openai_request_shapes():
    class Obj:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    sent = {}

    class Completions:
        def create(self, **kw):
            sent.update(kw)
            return Obj(
                choices=[Obj(message=Obj(content="Answer [P1]"))], usage=Obj(prompt_tokens=5, completion_tokens=2)
            )

    chat = llm.OpenAILLM.__new__(llm.OpenAILLM)
    chat.client = Obj(chat=Obj(completions=Completions()))
    chat.model, chat.temperature, chat.max_tokens, chat.name = "gpt-4o-mini", 0.2, 64, "openai/gpt-4o-mini"
    out = chat.generate("sys", "user")
    assert sent["model"] == "gpt-4o-mini" and sent["messages"][1] == {"role": "user", "content": "user"}
    assert out.output_tokens == 2
