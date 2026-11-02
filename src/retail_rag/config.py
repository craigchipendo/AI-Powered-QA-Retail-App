"""Runtime configuration.

All settings come from environment variables (or a local `.env` file), never
from code. Every variable is prefixed with `RAG_` except the provider SDKs' own
standard names (`OPENAI_API_KEY`, `GOOGLE_CLOUD_PROJECT`, ...).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    # ----------------------------------------------------------- providers
    embedding_provider: Literal["hashing", "local", "vertex", "openai", "ollama"] = "hashing"
    embedding_model: str | None = None  # provider default when unset
    embedding_dim: int = 384  # only used by the hashing embedder
    embedding_batch_size: int = 32

    llm_provider: Literal["offline", "vertex", "openai", "ollama"] = "offline"
    llm_model: str | None = None
    llm_temperature: float = 0.2
    llm_max_output_tokens: int = 512
    llm_timeout_s: float = 30.0

    # ------------------------------------------------------- vector store
    vector_store: Literal["memory", "postgres"] = "memory"
    database_url: SecretStr = SecretStr("postgresql://rag:rag@localhost:5432/retail")
    db_pool_min: int = 1
    db_pool_max: int = 10
    memory_store_path: Path = PROJECT_ROOT / ".cache" / "memory_index"

    # ------------------------------------------------------------- data
    catalogue_csv: Path = PROJECT_ROOT / "data" / "raw" / "retail_toy_dataset.csv"
    chunk_size: int = 600
    chunk_overlap: int = 80

    # -------------------------------------------------------- retrieval
    candidate_k: int = 40  # per retriever, before fusion
    max_products: int = 5  # products passed to the LLM
    chunks_per_product: int = 2
    context_char_budget: int = 6000
    # A retrieved product counts as evidence when EITHER gate passes; with no evidence the
    # service refuses without calling the LLM. Calibrate both with `retail-rag eval`.
    min_similarity: float | None = None  # vector gate; None = the embedding provider's default
    min_query_coverage: float = 0.55  # lexical gate: share of query terms found in the product text
    hybrid_vector_weight: float = 1.0
    hybrid_keyword_weight: float = 1.0
    rrf_k: int = 60
    search_mode: Literal["hybrid", "vector", "keyword"] = "hybrid"

    # ------------------------------------------------------------- api
    api_key: SecretStr | None = None  # when set, /v1/* requires X-API-Key
    max_query_chars: int = 500
    cors_origins: list[str] = Field(default_factory=list)
    log_level: str = "INFO"
    log_json: bool = True

    # ------------------------------------------------- provider specifics
    gcp_project: str | None = Field(default=None, validation_alias="GOOGLE_CLOUD_PROJECT")
    gcp_location: str = Field(default="us-central1", validation_alias="GOOGLE_CLOUD_LOCATION")
    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, validation_alias="OPENAI_BASE_URL")
    ollama_base_url: str = Field(default="http://localhost:11434", validation_alias="OLLAMA_BASE_URL")

    prompts_dir: Path = PROJECT_ROOT / "prompts"


@lru_cache
def get_settings() -> Settings:
    return Settings()
