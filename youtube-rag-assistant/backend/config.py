"""
Centralized application configuration.

All tunables live here and are overridable via environment variables or a
`.env` file (see `.env.example`). Using pydantic-settings keeps validation
and defaults in one place instead of scattering `os.getenv()` calls.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- LLM (Groq — free tier, no local GPU required) ----------------------
    groq_api_key: str = Field(default="", description="Get a free key at https://console.groq.com")
    groq_model: str = Field(
        default="openai/gpt-oss-120b",
        description="Free-tier Groq chat model. See https://console.groq.com/docs/models",
    )
    llm_temperature: float = 0.2

    # --- Embeddings (fully local, free, no API key) --------------------------
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"

    # --- Chunking --------------------------------------------------------------
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # --- Retrieval ---------------------------------------------------------------
    default_top_k: int = 4

    # --- Storage -------------------------------------------------------------------
    vector_store_dir: Path = Path("data/vector_store")

    # --- API / CORS ------------------------------------------------------------------
    cors_origins: list[str] = ["*"]
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
settings.vector_store_dir.mkdir(parents=True, exist_ok=True)
