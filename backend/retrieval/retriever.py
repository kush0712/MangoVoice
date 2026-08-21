"""
MangoVoice — Hybrid retriever.

dense top-20 + BM25 top-20 → RRF → top-8 final candidates.
Also builds the RetrievalResult with confidence scoring.
"""
from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

from backend.config import settings
from backend.schemas import RetrievalResult
from backend.retrieval.embeddings import embed_query
from backend.retrieval.lancedb_store import get_store
from backend.retrieval.fusion import rrf_fuse
from backend.retrieval.confidence import compute_retrieval_confidence
from backend.telemetry import get_logger

logger = get_logger(__name__)

# Dedicated 2-worker pool for LanceDB I/O.
# Using None (default executor) causes thread pool starvation under sequential
# load on Railway's 1-vCPU container — P70 spikes to 700ms+.
# A scoped pool with exactly 2 workers (dense + BM25 run concurrently) means
# retrieval threads are never starved by other executor work in the process.
_RETRIEVAL_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="lancedb")

# Process-local retrieval cache keyed by normalised query string.
# The LanceDB index is read-only at runtime, so results are deterministic
# for the process lifetime. Mirrors config.retrieval_cache_size.
_retrieval_cache: dict[str, RetrievalResult] = {}
_CACHE_MAXSIZE = 128


async def hybrid_retrieve(query: str) -> tuple[RetrievalResult, float]:
    """
    Full hybrid retrieval pipeline with process-local LRU cache.

    Cache hit:  ~0.2ms (dict lookup, no IO).
    Cache miss: embed (~30-80ms) + dense + BM25 concurrent (~150-250ms).
    Returns (RetrievalResult, embedding_ms).
    """
    t_total = time.perf_counter()

    # ── Cache hit fast path ───────────────────────────────────────────────────
    if query in _retrieval_cache:
        total_ms = (time.perf_counter() - t_total) * 1000
        logger.info(
            "Retrieval cache HIT: n_sources=%d",
            len(_retrieval_cache[query].sources),
            extra={"stage": "retrieval", "latency_ms": round(total_ms, 1)},
        )
        return _retrieval_cache[query], 0.0

    store = get_store()

    # ── Embed query ───────────────────────────────────────────────────────────
    t_embed = time.perf_counter()
    query_vec = embed_query(query)
    embedding_ms = (time.perf_counter() - t_embed) * 1000

    # ── Dense + BM25 concurrently (cache miss: full IO) ───────────────────────
    loop = asyncio.get_running_loop()
    dense_task = loop.run_in_executor(
        _RETRIEVAL_EXECUTOR, store.dense_search, query_vec, settings.dense_top_k
    )
    bm25_task = loop.run_in_executor(
        _RETRIEVAL_EXECUTOR, store.bm25_search, query, settings.bm25_top_k
    )
    dense_results, bm25_results = await asyncio.gather(dense_task, bm25_task)

    # ── Fuse + confidence ─────────────────────────────────────────────────────
    fused = rrf_fuse(dense_results, bm25_results, top_k=settings.final_top_k)
    result = compute_retrieval_confidence(
        sources=fused,
        dense_sources=dense_results,
        bm25_sources=bm25_results,
    )

    # ── Store in cache (simple LRU: evict oldest when full) ───────────────────
    if len(_retrieval_cache) >= _CACHE_MAXSIZE:
        _retrieval_cache.pop(next(iter(_retrieval_cache)))
    _retrieval_cache[query] = result

    total_ms = (time.perf_counter() - t_total) * 1000
    logger.info(
        "Hybrid retrieval: top_score=%.3f confidence=%.3f n_sources=%d",
        result.top_score, result.confidence, len(result.sources),
        extra={"stage": "retrieval", "latency_ms": round(total_ms, 1)},
    )
    return result, embedding_ms
