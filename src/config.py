"""Centralised application settings backed by a `.env` file.

pydantic-settings v2 automatically maps each field name to an upper-cased
environment variable (e.g. ``chunk_size`` → ``CHUNK_SIZE``).  Every setting
can be overridden at runtime by exporting the corresponding variable or by
adding it to the ``.env`` file — no code changes required.
"""

import os
from pathlib import Path

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict

    class Settings(BaseSettings):
        """Application settings.  All fields are readable from environment
        variables or ``.env`` (case-insensitive, converted to UPPER_SNAKE_CASE
        automatically by pydantic-settings v2).
        """

        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            # Silently ignore .env fields that don't map to a declared setting.
            extra="ignore",
        )

        # ── Chunking ──────────────────────────────────────────────────────────
        chunk_size: int = 500
        chunk_overlap: int = 50

        # ── Retrieval ─────────────────────────────────────────────────────────
        top_k: int = 4

        # ── LLM model names ───────────────────────────────────────────────────
        openai_model: str = "gpt-3.5-turbo"
        # NOTE: "gemini-pro" was deprecated by Google in Feb 2025.
        #       Default updated to "gemini-1.5-flash" (stable, low-latency).
        google_model: str = "gemini-1.5-flash"
        hf_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
        # FIX 6: cross-encoder model is now a configurable setting, no longer
        # hardcoded in retrieval.py.  Override via CROSS_ENCODER_MODEL in .env.
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

        # ── API keys (loaded from .env) ────────────────────────────────────────
        openai_api_key: str = ""
        google_api_key: str = ""

        # ── Path locations ────────────────────────────────────────────────────
        data_dir: Path = Path("data/sample_docs")
        logs_dir: Path = Path("logs")
        vectorstore_dir: Path = Path("data/vectorstore")
        faiss_index_dir: Path = Path("data/faiss_index")

    settings = Settings()

except ImportError:
    # Fallback to os.getenv if pydantic-settings is not installed
    class _Settings:
        chunk_size: int = int(os.getenv("CHUNK_SIZE", 500))
        chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", 50))
        top_k: int = int(os.getenv("TOP_K", 4))

        openai_model: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        # NOTE: "gemini-pro" was deprecated by Google in Feb 2025.
        google_model: str = os.getenv("GOOGLE_MODEL", "gemini-1.5-flash")
        hf_embedding_model: str = os.getenv(
            "HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        # FIX 6: cross-encoder model now configurable via env var.
        cross_encoder_model: str = os.getenv(
            "CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

        openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
        google_api_key: str = os.getenv("GOOGLE_API_KEY", "")

        data_dir: Path = Path(os.getenv("DATA_DIR", "data/sample_docs"))
        logs_dir: Path = Path(os.getenv("LOGS_DIR", "logs"))
        vectorstore_dir: Path = Path(os.getenv("VECTORSTORE_DIR", "data/vectorstore"))
        faiss_index_dir: Path = Path(os.getenv("FAISS_INDEX_DIR", "data/faiss_index"))

    settings = _Settings()
