"""
MangoVoice — LanceDB embedded vector + BM25 store.

Opens a pre-built read-only index at startup.
Provides: dense vector search, BM25 full-text search.
"""
from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np

from backend.config import settings
from backend.schemas import RetrievalSource, ChunkingStrategy
from backend.telemetry import get_logger

logger = get_logger(__name__)

_store_singleton: Optional["LanceDBStore"] = None


class LanceDBStore:
    """Embedded LanceDB store for MangoVoice."""

    def __init__(self) -> None:
        self._db = None
        self._table = None
        self._ready = False

    def _load(self) -> None:
        if self._table is not None:
            return

        index_path = settings.index_path
        if not os.path.exists(index_path):
            logger.warning(
                "LanceDB index not found at %s — running in demo/stub mode", index_path
            )
            self._ready = False
            return

        try:
            import lancedb

            self._db = lancedb.connect(index_path)
            self._table = self._db.open_table(settings.index_table)
            self._ready = True
            logger.info(
                "LanceDB index opened: %s / %s (%d rows)",
                index_path,
                settings.index_table,
                self._table.count_rows(),
            )
        except Exception as exc:
            logger.error("Failed to open LanceDB index: %s", exc)
            self._ready = False

    def is_ready(self) -> bool:
        return self._ready

    def dense_search(self, query_vec: np.ndarray, top_k: int) -> list[RetrievalSource]:
        """ANN vector search. Returns top_k candidates."""
        if not self._ready:
            return _demo_sources(top_k)

        t0 = time.perf_counter()
        try:
            results = (
                self._table.search(query_vec.tolist())
                .metric("cosine")
                .limit(top_k)
                .to_list()
            )
        except Exception as exc:
            logger.error("Dense search failed: %s", exc)
            return []

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            "Dense search: %d results", len(results),
            extra={"stage": "dense_search", "latency_ms": round(latency_ms, 1)},
        )
        return [_row_to_source(r, rank=i) for i, r in enumerate(results)]

    def bm25_search(self, query_text: str, top_k: int) -> list[RetrievalSource]:
        """BM25 full-text search. Returns top_k candidates."""
        if not self._ready:
            return _demo_sources(top_k)

        t0 = time.perf_counter()
        try:
            results = (
                self._table.search(query_text, query_type="fts")
                .limit(top_k)
                .to_list()
            )
        except Exception as exc:
            logger.error("BM25 search failed: %s", exc)
            return []

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            "BM25 search: %d results", len(results),
            extra={"stage": "bm25_search", "latency_ms": round(latency_ms, 1)},
        )
        return [_row_to_source(r, rank=i) for i, r in enumerate(results)]


def _row_to_source(row: dict, rank: int = 0) -> RetrievalSource:
    score = float(row.get("_distance", 0.0))
    # Cosine distance → similarity: similarity = 1 - distance
    if score <= 1.0:
        score = max(0.0, 1.0 - score)

    strategy_str = row.get("strategy", "parent_child")
    try:
        strategy = ChunkingStrategy(strategy_str)
    except ValueError:
        strategy = ChunkingStrategy.PARENT_CHILD

    return RetrievalSource(
        chunk_id=str(row.get("chunk_id", f"chunk-{rank}")),
        parent_id=str(row.get("parent_id", "")),
        score=score,
        raw_dense_score=score,  # preserve cosine similarity before RRF overwrites .score
        dense_rank=rank,
        text=str(row.get("text", "")),
        language=str(row.get("language", "en")),
        strategy=strategy,
        query_id=str(row.get("query_id", "")) or None,
    )


def _demo_sources(n: int) -> list[RetrievalSource]:
    """Stub sources returned when no index is available (demo mode)."""
    demo_texts = [
        "The Manhattan Project was a research and development undertaking during World War II that produced the first nuclear weapons.",
        "India gained independence from British rule on August 15, 1947, following a long independence movement led by figures like Mahatma Gandhi.",
        "The Goa Liberation was completed on December 19, 1961, when Indian Armed Forces ended Portuguese colonial rule.",
        "Artificial intelligence refers to the simulation of human intelligence in machines programmed to think and learn.",
        "Retrieval-augmented generation (RAG) combines information retrieval with language model generation for grounded answers.",
    ]
    return [
        RetrievalSource(
            chunk_id=f"demo-chunk-{i}",
            parent_id=f"demo-parent-{i}",
            score=0.85 - i * 0.05,
            dense_rank=i,
            text=demo_texts[i % len(demo_texts)],
            language="en",
            strategy=ChunkingStrategy.PARENT_CHILD,
        )
        for i in range(min(n, len(demo_texts)))
    ]


def get_store() -> LanceDBStore:
    global _store_singleton
    if _store_singleton is None:
        _store_singleton = LanceDBStore()
        _store_singleton._load()
    return _store_singleton
