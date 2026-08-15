"""
MangoVoice — Application configuration via pydantic-settings.
All values can be overridden via environment variables or .env file.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env.local",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── API keys (server-side only) ──────────────────────────────────────────
    sarvam_api_key: str = ""
    groq_api_key: str = ""

    # ── Model config ─────────────────────────────────────────────────────────
    groq_model: str = "llama-3.1-8b-instant"
    groq_safety_model: str = "llama-3.1-8b-instant"  # fast safety eval
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    sarvam_model: str = "saaras:v3"
    sarvam_mode: str = "codemix"

    # ── Index / data ─────────────────────────────────────────────────────────
    index_path: str = "data/lancedb"
    index_version: str = "v1"
    index_table: str = "chunks"

    # ── App limits ───────────────────────────────────────────────────────────
    max_audio_seconds: int = 25
    max_audio_bytes: int = 10 * 1024 * 1024  # 10 MB
    max_transcript_chars: int = 1000
    max_output_tokens: int = 512

    # ── Retrieval config ─────────────────────────────────────────────────────
    dense_top_k: int = 20
    bm25_top_k: int = 20
    final_top_k: int = 8
    rrf_k: int = 60

    # ── Confidence gate (calibrated for consistent mean-pooling multilingual MiniLM) ─
    confidence_low_threshold: float = 0.18   # allow English/Hindi/Hinglish queries through (genuine semantic overlap)
    confidence_margin_min: float = 0.05
    confidence_min_supporting: int = 2

    # ── Timeouts (seconds) ───────────────────────────────────────────────────
    sarvam_timeout: float = 12.0
    groq_timeout: float = 8.0
    groq_safety_timeout: float = 3.0

    # ── Cache ────────────────────────────────────────────────────────────────
    embedding_cache_size: int = 256
    retrieval_cache_size: int = 128

    # ── Misc ─────────────────────────────────────────────────────────────────
    app_env: str = "development"
    log_level: str = "INFO"
    demo_mode: bool = False  # returns mock responses when True (no API keys needed)

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def has_sarvam_key(self) -> bool:
        return bool(self.sarvam_api_key)

    @property
    def has_groq_key(self) -> bool:
        return bool(self.groq_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


# Module-level singleton for convenient import
settings: Settings = get_settings()
