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
    groq_model: str = "allam-2-7b"                  # fast (sub-200ms), high accuracy grounded generation
    groq_safety_model: str = "meta-llama/llama-prompt-guard-2-22m"   # fast safety eval
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
    # Balanced recall/latency: top-20 gives >98% recall at 63k scale while
    # halving ANN partition scans vs top-50. Extractive needs only top-10 after RRF.
    dense_top_k: int = 20
    bm25_top_k: int = 20
    final_top_k: int = 10   # chunks passed to extractive — top-10 is sufficient
    rrf_k: int = 60

    # ── Confidence gate ────────────────────────────────────────────────────────
    # Relaxed to 0.50. The extractive fallback now performs a strict semantic
    # verification of the specific candidate sentence (sim >= 0.55). This gate
    # only needs to drop completely unrelated junk, ensuring legitimate but
    # borderline cross-lingual matches (like Hinglish) reach the fallback.
    confidence_low_threshold: float = 0.50
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
