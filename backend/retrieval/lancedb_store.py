"""
MangoVoice — LanceDB embedded vector + BM25 store.

Opens a pre-built read-only index at startup.
Provides: dense vector search (in-memory numpy exact search), BM25 full-text search.

Dense search strategy
─────────────────────
At startup _load_into_ram() reads all 63,615 vectors into a pre-normalised
(N, 384) float32 numpy matrix (~98 MB). Every subsequent dense query is a
single matrix-vector dot product: ~2-3 ms with no disk IO and exact cosine
similarity. This avoids two problems with the IVF_PQ index:
  1. Disk IO latency (100+ ms cold) — data is already in RAM.
  2. PQ approximation error in _distance — score is the true cosine sim,
     so the confidence gate and extractive rules always see correct values.
If the RAM load fails (OOM etc.), dense_search falls back to LanceDB ANN.
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
        # In-memory exact search index (populated by _load_into_ram)
        self._vec_normed: Optional[np.ndarray] = None   # (N, 384) L2-normalised float32
        self._meta: Optional[list[dict]] = None          # parallel metadata per row

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
            return

        # Load all vectors into RAM for fast exact in-memory search
        self._load_into_ram()

    def _load_into_ram(self) -> None:
        """
        Read all 384-dim vectors into a pre-normalised (N, 384) numpy matrix.

        One-time startup cost (~2s to read ~98 MB from disk).  After this every
        dense_search call is a 2-3 ms matrix-vector dot product with no disk IO.
        """
        try:
            n = self._table.count_rows()
            logger.info("Loading %d vectors into RAM (~%.0f MB)...", n, n * 384 * 4 / 1e6)
            t0 = time.perf_counter()

            # Read all rows as an Arrow table (memory-efficient columnar format)
            arrow = self._table.to_arrow()

            # Build (N, 384) float32 and L2-normalise each row so cosine similarity
            # = dot product (avoids per-query norm computation)
            vecs = np.array(arrow["vector"].to_pylist(), dtype=np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            self._vec_normed = (vecs / norms).astype(np.float32)

            # Parallel metadata list — one dict per row, same order as _vec_normed
            _META_KEYS = ["chunk_id", "parent_id", "query_id", "language",
                          "strategy", "chunk_start", "chunk_end", "text"]
            cols = {k: arrow[k].to_pylist() for k in _META_KEYS}
            self._meta = [{k: cols[k][i] for k in _META_KEYS} for i in range(n)]

            elapsed = (time.perf_counter() - t0) * 1000
            logger.info(
                "In-memory vector index ready: %d vectors, %.1f MB, loaded in %.0f ms",
                n, self._vec_normed.nbytes / 1e6, elapsed,
            )
        except Exception as exc:
            logger.warning(
                "Failed to load vectors into RAM — dense search will use LanceDB ANN fallback: %s", exc
            )
            self._vec_normed = None
            self._meta = None

    def is_ready(self) -> bool:
        return self._ready

    def dense_search(self, query_vec: np.ndarray, top_k: int) -> list[RetrievalSource]:
        """Exact cosine search. Uses in-memory numpy if available, LanceDB ANN as fallback."""
        if not self._ready:
            return _demo_sources(top_k)

        if self._vec_normed is not None:
            return self._dense_numpy(query_vec, top_k)

        # Fallback: IVF_PQ ANN (used only if _load_into_ram failed)
        return self._dense_lancedb(query_vec, top_k)

    def _dense_numpy(self, query_vec: np.ndarray, top_k: int) -> list[RetrievalSource]:
        """
        2-3 ms exact cosine search via pre-normalised matrix dot product.
        raw_dense_score is the true cosine similarity — no PQ approximation.
        """
        t0 = time.perf_counter()

        norm_q = np.linalg.norm(query_vec)
        if norm_q == 0:
            return []
        q_normed = (query_vec / norm_q).astype(np.float32)

        # Single BLAS matrix-vector multiply: (N, 384) @ (384,) → (N,) cosine sims
        sims = self._vec_normed @ q_normed

        # argpartition is O(N); then sort only the top-k slice
        top_k = min(top_k, len(sims))
        top_idx = np.argpartition(sims, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]

        sources = []
        for rank, idx in enumerate(top_idx):
            meta = self._meta[idx]
            sim = float(sims[idx])
            strategy_str = meta.get("strategy") or "parent_child"
            try:
                strategy = ChunkingStrategy(strategy_str)
            except ValueError:
                strategy = ChunkingStrategy.PARENT_CHILD

            sources.append(RetrievalSource(
                chunk_id=str(meta["chunk_id"]),
                parent_id=str(meta["parent_id"] or ""),
                score=sim,
                raw_dense_score=sim,   # exact — used by confidence gate + extractive rules
                dense_rank=rank,
                text=str(meta["text"] or ""),
                language=str(meta["language"] or "en"),
                strategy=strategy,
                query_id=str(meta.get("query_id") or "") or None,
            ))

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            "Dense search (numpy): %d results in %.2f ms", len(sources), latency_ms,
            extra={"stage": "dense_search", "latency_ms": round(latency_ms, 1)},
        )
        return sources

    def _dense_lancedb(self, query_vec: np.ndarray, top_k: int) -> list[RetrievalSource]:
        """
        Fallback IVF_PQ ANN search via LanceDB (used only if RAM load failed).
        Recomputes exact cosine from stored vectors to correct PQ approximation error.
        """
        t0 = time.perf_counter()
        try:
            results = (
                self._table.search(query_vec.tolist())
                .metric("cosine")
                .nprobes(8)
                .limit(top_k)
                .to_list()
            )
        except Exception as exc:
            logger.error("LanceDB dense search failed: %s", exc)
            return []

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            "Dense search (LanceDB fallback): %d results", len(results),
            extra={"stage": "dense_search", "latency_ms": round(latency_ms, 1)},
        )

        # Correct PQ approximation: recompute exact cosine from stored float32 vectors
        sources = []
        norm_q = np.linalg.norm(query_vec)  # constant — compute once
        for i, r in enumerate(results):
            source = _row_to_source(r, rank=i)
            if "vector" in r and norm_q > 0:
                v_stored = np.array(r["vector"], dtype=np.float32)
                norm_v = np.linalg.norm(v_stored)
                if norm_v > 0:
                    true_sim = float(np.dot(query_vec, v_stored) / (norm_q * norm_v))
                    source.raw_dense_score = true_sim
                    source.score = true_sim
            sources.append(source)

        sources.sort(key=lambda s: s.raw_dense_score, reverse=True)
        for i, s in enumerate(sources):
            s.dense_rank = i
        return sources

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

