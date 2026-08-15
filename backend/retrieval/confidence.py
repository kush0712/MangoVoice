"""
MangoVoice — Retrieval confidence gate.

Decides PASS or REFUSE based on:
1. top candidate score
2. top-1 vs top-2 margin
3. number of independent supporting candidates
4. dense / BM25 agreement
5. (optional) parent-level score aggregation

Threshold is calibrated via evaluation/threshold_calibration.py.
"""
from __future__ import annotations

from backend.config import settings
from backend.schemas import RetrievalResult, RetrievalSource
from backend.telemetry import get_logger

logger = get_logger(__name__)


def compute_retrieval_confidence(
    sources: list[RetrievalSource],
    dense_sources: list[RetrievalSource] | None = None,
    bm25_sources: list[RetrievalSource] | None = None,
) -> RetrievalResult:
    """
    Build a RetrievalResult with composite confidence from fused sources.

    Confidence is driven primarily by the raw cosine similarity of the top
    dense result — NOT the tiny RRF scores (which are always near 0.016 by
    design and carry almost no absolute relevance signal).
    """
    if not sources:
        return RetrievalResult(
            sources=[],
            top_score=0.0,
            margin=0.0,
            confidence=0.0,
            supporting_count=0,
        )

    top_score = sources[0].score
    second_score = sources[1].score if len(sources) > 1 else 0.0
    margin = top_score - second_score

    # ── Primary signal: raw cosine similarity from top dense result ──────────
    # raw_dense_score is the preserved cosine similarity (1 - distance) set by
    # lancedb_store before RRF fusion overwrites .score with the tiny RRF value.
    dense_top_sim = dense_sources[0].raw_dense_score if dense_sources else 0.0

    # Calibrated thresholds for multilingual MiniLM (paraphrase variant, mean pooling):
    # >0.25  → strong semantic match  (norm = 1.0)
    # 0.08–0.25 → moderate/paraphrase overlap  (norm scales linearly)
    # <0.08  → likely keyword coincidence or fully off-topic (norm ≈ 0)
    # These values work for English, Hindi, and Hinglish with consistent mean-pooling embeddings.
    SIM_HIGH = 0.25
    SIM_LOW  = 0.08
    if dense_top_sim >= SIM_HIGH:
        norm_dense = 1.0
    elif dense_top_sim <= SIM_LOW:
        norm_dense = 0.0
    else:
        norm_dense = (dense_top_sim - SIM_LOW) / (SIM_HIGH - SIM_LOW)

    # ── Cross-modal bonus: dense AND BM25 agree on top chunk ─────────────────
    dense_top_id = dense_sources[0].chunk_id if dense_sources else None
    bm25_top_id  = bm25_sources[0].chunk_id  if bm25_sources  else None
    agree = (dense_top_id is not None and dense_top_id == bm25_top_id)

    supporting = len(sources)

    # ── Composite confidence ─────────────────────────────────────────────────
    # 70% raw semantic similarity, 20% cross-modal agreement, 10% count bonus
    raw_conf = (
        norm_dense * 0.70
        + (0.20 if agree else 0.0)
        + min(supporting / 10, 1.0) * 0.10
    )
    confidence = min(max(raw_conf, 0.0), 1.0)

    return RetrievalResult(
        sources=sources,
        top_score=top_score,
        margin=margin,
        confidence=confidence,
        dense_bm25_agree=agree,
        supporting_count=supporting,
    )


def should_generate(result: RetrievalResult) -> bool:
    """
    PASS/REFUSE decision from calibrated thresholds.
    Returns True → proceed to generation; False → REFUSE.
    """
    T_low = settings.confidence_low_threshold

    if result.confidence < T_low:
        logger.info(
            "Confidence gate: REFUSE (confidence=%.3f < T_low=%.3f)",
            result.confidence, T_low,
        )
        return False

    # Borderline safety: only require cross-modal agreement in a tighter band
    # (T_low → T_low * 1.20) to avoid keyword coincidences slipping through,
    # but with a wider tolerance than before so legitimate English queries aren't refused.
    is_borderline = result.confidence < T_low * 1.20
    if is_borderline and not result.dense_bm25_agree and result.supporting_count < 2:
        logger.info(
            "Confidence gate: REFUSE borderline without cross-modal agreement and < 2 sources (confidence=%.3f agree=%s)",
            result.confidence, result.dense_bm25_agree,
        )
        return False

    return True
