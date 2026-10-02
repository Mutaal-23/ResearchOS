"""Application configuration.

Every tunable in ResearchOS is declared here exactly once, with a type, a
default, and a validation rule. Nothing else in the codebase calls
``os.getenv`` - if a setting is not on this object, it does not exist.

Why bother over ``os.getenv()``? Because ``os.getenv`` fails silently.
Misspell a variable name and you get ``None`` with no error, which usually
surfaces much later as a confusing 401 or a connection error. Here, a missing
or invalid value raises at *import time*, so the app refuses to boot rather
than failing halfway through a user's request.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# src/researchos/config.py -> src/researchos -> src -> <project root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Validated application settings, populated from environment and .env."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        # Names arrive from .env as free text; pydantic coerces to the declared
        # type (int, float, bool) and raises on a value it cannot parse.
        extra="ignore",
    )

    # --- LLM ---------------------------------------------------------------
    gemini_api: str = Field(..., min_length=1, description="Google Gemini API key.")

    gemini_models: Annotated[list[str], NoDecode] = Field(
        # Ordered fallback chain, comma separated. The client tries each name in
        # turn and moves on when the provider reports the model as missing or
        # rate limited.
        #
        # Google's catalogue rotates and retired models start returning 404
        # rather than a redirect, so a chain built once goes stale quietly.
        # Listed newest-stable first, then the moving alias as a last resort:
        # gemini-2.5-flash and gemini-2.0-flash both now 404.
        default_factory=lambda: ["gemini-3.8-flash", "gemini-3.5-flash", "gemini-flash-latest"],
        description=(
            "Ordered fallback chain, comma separated. The client tries each name "
            "in turn and moves on when the provider reports the model as "
            "missing or rate limited."
        ),
    )

    llm_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Keep low for grounded answers. High temperature invents facts.",
    )

    llm_max_tokens: int = Field(default=2048, ge=64, le=32768)

    # --- PostgreSQL --------------------------------------------------------
    postgres_user: str = "researchos"
    postgres_password: str = "researchos"
    postgres_db: str = "researchos"
    postgres_host: str = "localhost"
    postgres_port: int = Field(default=5433, ge=1, le=65535)

    # --- Qdrant ------------------------------------------------------------
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = "researchos-dev-key"
    qdrant_collection: str = "researchos_chunks"

    # --- Embeddings / reranking -------------------------------------------
    # Dense embeddings come from the Gemini API rather than a local ONNX
    # model. Hugging Face rate-limits this host (HTTP 429), and a hosted
    # embedding endpoint keeps the pipeline identical - Qdrant stores a
    # float vector regardless of who produced it.
    embedding_model: str = "gemini-embedding-001"

    # Sparse vectors are computed by our own BM25 implementation
    # (retrieval/bm25.py). It needs no model at all, which is both faster
    # and a better way to actually understand what BM25 does.
    sparse_model: str = "bm25-local"

    embedding_dim: int = Field(
        default=768,
        ge=1,
        description="Must match the output width of embedding_model. Changing it "
        "requires recreating the Qdrant collection and re-embedding "
        "everything already indexed.",
    )

    # --- Chunking ----------------------------------------------------------
    chunk_size_tokens: int = Field(default=400, ge=64, le=4096)
    chunk_overlap_tokens: int = Field(default=80, ge=0, le=2048)
    min_chunk_chars: int = Field(default=80, ge=0)

    # --- Retrieval ---------------------------------------------------------
    retrieval_candidates: int = Field(
        default=40,
        ge=1,
        le=500,
        description="Candidates fetched from each retriever before fusion.",
    )
    retrieval_top_k: int = Field(default=6, ge=1, le=100)
    rrf_k: int = Field(
        default=60,
        ge=1,
        description="Reciprocal Rank Fusion damping constant. From the original "
        "Cormack et al. paper; higher values flatten rank differences.",
    )

    # --- Paths -------------------------------------------------------------
    eval_questions_path: Path = PROJECT_ROOT / "evals" / "questions.json"
    migrations_path: Path = PROJECT_ROOT / "migrations"

    # --- Server ------------------------------------------------------------
    app_host: str = "127.0.0.1"
    app_port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = "INFO"

    @field_validator("gemini_models", mode="before")
    @classmethod
    def _split_model_list(cls, value: object) -> object:
        """Accept the comma-separated .env form and turn it into a list.

        ``NoDecode`` above is what actually makes this reachable. Without it the
        settings source runs ``json.loads`` on the raw string first, and a
        comma-separated value is not valid JSON, so this validator is never
        called. The two pieces have to be changed together.
        """
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("chunk_overlap_tokens")
    @classmethod
    def _overlap_must_be_smaller_than_chunk(cls, overlap: int, info: object) -> int:
        """Reject overlap >= size.

        Without this, an overlap at or above the chunk size produces an
        infinite loop: each window starts at or before the previous window's
        end, so the offset never advances and ingestion never terminates.
        """
        data = getattr(info, "data", {})
        size = data.get("chunk_size_tokens")
        if size is not None and overlap >= size:
            raise ValueError(
                f"chunk_overlap_tokens ({overlap}) must be less than "
                f"chunk_size_tokens ({size}), otherwise chunking cannot advance."
            )
        return overlap

    @property
    def database_url(self) -> str:
        """Assemble the Postgres connection URL.

        Uses the keyword-argument form rather than an f-string so that special
        characters in the password (a common source of "why won't this
        connect?") are percent-encoded correctly by the driver.
        """
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def database_url_safe(self) -> str:
        """Connection URL with the password redacted, safe for logging."""
        return (
            f"postgresql://{self.postgres_user}:***"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    ``lru_cache`` means the .env file is read and validated exactly once, on
    first access, no matter how many times settings are requested. Building
    ``Settings()`` is not free - it re-reads and re-parses .env every time -
    so this must not be called per request.
    """
    return Settings()
