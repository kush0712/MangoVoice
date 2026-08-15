"""
MangoVoice — Reciprocal Rank Fusion.

RRF(d) = Σ 1 / (k + rank_m(d))   where k=60 (conventional)

Fuses dense and BM25 ranked lists into a single ranked list.
"""
from __future__ import annotations

from backend.schemas import RetrievalSource
from backend.config import settings


def rrf_fuse(
    dense_results: list[RetrievalSource],
    bm25_results: list[RetrievalSource],
    top_k: int | None = None,
) -> list[RetrievalSource]:
    """
    Fuse two ranked lists via Reciprocal Rank Fusion.
    Returns deduplicated, re-ranked results.
    """
    k = settings.rrf_k
    top_k = top_k or settings.final_top_k

    # Build RRF score accumulator indexed by chunk_id
    scores: dict[str, float] = {}
    sources_map: dict[str, RetrievalSource] = {}

    for rank, src in enumerate(dense_results):
        cid = src.chunk_id
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        src.dense_rank = rank
        sources_map[cid] = src

    for rank, src in enumerate(bm25_results):
        cid = src.chunk_id
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        src.bm25_rank = rank
        if cid not in sources_map:
            sources_map[cid] = src

    # Sort by RRF score descending
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    result = []
    for cid, rrf_score in ranked:
        src = sources_map[cid]
        src.rrf_score = rrf_score
        # Use RRF score as the primary score for downstream confidence gate
        src.score = rrf_score
        result.append(src)

    return result
