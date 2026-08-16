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
        # Accept .env.local for local dev; in production (Railway) env vars are
        # injected directly — pydantic-settings reads them without a file.
        env_file=(".env.local", ".env"),
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
    # startup.sh exports INDEX_PATH pointing to Railway persistent volume;
    # falls back to local path for development.
    index_path: str = "data/lancedb"
    fastembed_cache_dir: str = "data/fastembed_cache"
    index_version: str = "v1"
    index_table: str = "chunks"

    # ── App limits ───────────────────────────────────────────────────────────
    max_audio_seconds: int = 25
    max_audio_bytes: int = 10 * 1024 * 1024  # 10 MB
    max_transcript_chars: int = 512
    max_output_tokens: int = 512

    # ── Retrieval config ─────────────────────────────────────────────────────
    # Maximise recall: more candidates before RRF → better chance of surfacing
    # the correct chunk. LanceDB ANN+BM25 are fast enough to handle top-50.
    dense_top_k: int = 50
    bm25_top_k: int = 50
    final_top_k: int = 20   # chunks passed to extractive (was 8)
    rrf_k: int = 60

    # ── Confidence gate ────────────────────────────────────────────────────────
    # Calibrated to 0.75 for maximum precision. Since final_top_k is now 20,
    # confidence naturally inflated. 0.75 cleanly separates genuine dataset
    # hits (Gandhi, Blood Pressure, Diabetes -> 0.80+) from coincidental
    # keyword overlaps (Malaria, WW2 -> 0.60-0.69) where the dataset lacks
    # the answer. This guarantees we don't return wrong answers.
    confidence_low_threshold: float = 0.75
    confidence_margin_min: float = 0.05
    confidence_min_supporting: int = 2

    # ── Timeouts (seconds) ───────────────────────────────────────────────────
    sarvam_timeout: float = 12.0
    groq_timeout: float = 8.0
    groq_safety_timeout: float = 3.0

    # ── Cache ────────────────────────────────────────────────────────────────
    embedding_cache_size: int = 256
    retrieval_cache_size: int = 128

    # ── CORS ─────────────────────────────────────────────────────────────────
    # Comma-separated list of allowed origins, e.g. https://mangovoice.vercel.app
    # Defaults to "*" for development. Set via ALLOWED_ORIGINS env var in production.
    allowed_origins: str = "*"

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

    @property
    def cors_origins(self) -> list[str]:
        """Parse ALLOWED_ORIGINS into a list."""
        if self.allowed_origins == "*":
            return ["*"]
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


# Module-level singleton for convenient import
settings: Settings = get_settings()
