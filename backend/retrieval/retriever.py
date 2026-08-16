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


async def hybrid_retrieve(query: str) -> tuple[RetrievalResult, float]:
    """
    Full hybrid retrieval pipeline.
    Returns (RetrievalResult, embedding_ms).
    """
    t_total = time.perf_counter()
    store = get_store()

    # 1. Embed query (sync but fast)
    t_embed = time.perf_counter()
    query_vec = embed_query(query)
    embedding_ms = (time.perf_counter() - t_embed) * 1000

    # 2. Dense + BM25 search concurrently (both are sync but we use a thread pool
    #    to avoid blocking the event loop for large indexes)
    t_retrieve = time.perf_counter()

    loop = asyncio.get_running_loop()
    dense_task = loop.run_in_executor(
        _RETRIEVAL_EXECUTOR, store.dense_search, query_vec, settings.dense_top_k
    )
    bm25_task = loop.run_in_executor(
        _RETRIEVAL_EXECUTOR, store.bm25_search, query, settings.bm25_top_k
    )
    dense_results, bm25_results = await asyncio.gather(dense_task, bm25_task)

    retrieval_ms = (time.perf_counter() - t_retrieve) * 1000

    # 3. Fuse via RRF
    fused = rrf_fuse(dense_results, bm25_results, top_k=settings.final_top_k)

    # 4. Build confidence result
    result = compute_retrieval_confidence(
        sources=fused,
        dense_sources=dense_results,
        bm25_sources=bm25_results,
    )

    total_ms = (time.perf_counter() - t_total) * 1000
    logger.info(
        "Hybrid retrieval: top_score=%.3f confidence=%.3f n_sources=%d",
        result.top_score, result.confidence, len(result.sources),
        extra={"stage": "retrieval", "latency_ms": round(total_ms, 1)},
    )
    return result, embedding_ms
