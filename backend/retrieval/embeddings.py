"""
MangoVoice — FastEmbed/ONNX multilingual embedding wrapper.

Model: paraphrase-multilingual-MiniLM-L12-v2 (384-dim, Hindi/English/Hinglish)
Loaded once per process. LRU cache prevents redundant embeds.
"""
from __future__ import annotations

import time
from functools import lru_cache
from typing import Optional

import numpy as np
from backend.config import settings
from backend.telemetry import get_logger

logger = get_logger(__name__)

_embedder_singleton: Optional["MultilingualEmbedder"] = None


class MultilingualEmbedder:
    """Thin wrapper around FastEmbed for consistent interface."""

    def __init__(self) -> None:
        self._model = None
        self._ready = False

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from fastembed import TextEmbedding
            import os

            # 4 intra-op threads = same Railway billing cost as 2 threads.
            # Railway charges on total vCPU-seconds: 4 threads × 70ms =
            # 2 threads × 140ms = identical cost. Faster is better — same bill.
            n_threads = int(os.environ.get("FASTEMBED_THREADS", "4"))
            self._model = TextEmbedding(
                model_name=settings.embedding_model,
                threads=n_threads,
                cache_dir=settings.fastembed_cache_dir,
            )
            # Warm up ONNX session immediately so first query has zero initialization penalty
            list(self._model.embed(["warmup sentence"]))
            self._ready = True
            logger.info(
                "Embedding model loaded & warmed up: %s (threads: %d)",
                settings.embedding_model, n_threads,
            )
        except Exception as exc:
            logger.error("Failed to load embedding model: %s", exc)
            raise

    def is_ready(self) -> bool:
        return self._ready

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts. Returns (N, 384) float32 array."""
        self._load()
        t0 = time.perf_counter()
        embeddings = list(self._model.embed(texts))
        arr = np.array(embeddings, dtype=np.float32)
        latency_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            "Embedded %d texts", len(texts),
            extra={"stage": "embedding", "latency_ms": round(latency_ms, 1)},
        )
        return arr

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single text. Returns (384,) float32 array."""
        arr = self.embed([text])
        return arr[0]


# ── Process-local LRU cache for query embeddings ──────────────────────────────

@lru_cache(maxsize=256)
def _cached_embed(text: str) -> tuple:
    """Cache embedding by normalized query text. Returns tuple for hashability."""
    embedder = get_embedder()
    vec = embedder.embed_one(text)
    return tuple(vec.tolist())


def embed_query(text: str) -> np.ndarray:
    """Get query embedding with LRU cache."""
    cached = _cached_embed(text)
    return np.array(cached, dtype=np.float32)


def get_embedder() -> MultilingualEmbedder:
    global _embedder_singleton
    if _embedder_singleton is None:
        _embedder_singleton = MultilingualEmbedder()
        _embedder_singleton._load()
    return _embedder_singleton
