"""
MangoVoice — Async RAG orchestrator (state machine).

States:
  RECEIVED → TRANSCRIBING → TRANSCRIBED
  → SAFETY_CHECK + RETRIEVAL (parallel)
  → CONFIDENCE_GATE
  │ REFUSE
  └ EXTRACTIVE_FAST_PATH → GROUNDING_CHECK (lightweight, ~0ms)
                            │ PASS → return ANSWERED (<200ms guaranteed)
                            └ FAIL → return REFUSED with sources[:3]
    [Groq fires as background task — result stored by request_id for polling]

Design goals:
  1. Extractive answer is the <200ms contractual response.
  2. Groq is NEVER awaited in /api/query — zero blocking LLM latency.
  3. Groq result is stored in _polish_store (TTL 60s) so the frontend can
     poll /api/query/result/{request_id} and swap in the LLM-enhanced answer.
  4. Every refused branch returns sources[:3] when evidence exists.
  5. Per-stage timing captured precisely.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import Optional

from backend.config import settings
from backend.schemas import (
    ConfidenceLevel,
    GenerationResult,
    LatencyMetrics,
    PipelineStatus,
    QueryResponse,
    RefusalReason,
)
from backend.guardrails import normalize_text, layer1_check, layer2_prompt_guard
from backend.retrieval.retriever import hybrid_retrieve
from backend.retrieval.confidence import should_generate
from backend.generation import generate
from backend.grounding import verify_grounding, verify_grounding_extractive
from backend.fallback.extractive import extractive_fallback
from backend.telemetry import get_logger, new_request_id

logger = get_logger(__name__)

# ── In-memory polish store (TTL 60 s) ────────────────────────────────────────
# Stores Groq-generated answers keyed by request_id so the frontend can poll
# /api/query/result/{request_id} and progressively enhance the extractive answer.
# No Redis needed — the dict is process-local and cleaned up by _sweep_store().
_polish_store: dict[str, dict] = {}
_STORE_TTL = 60.0  # seconds


def _sweep_store() -> None:
    """Remove expired entries (called opportunistically, not on every request)."""
    now = time.monotonic()
    expired = [k for k, v in _polish_store.items() if now - v["ts"] > _STORE_TTL]
    for k in expired:
        del _polish_store[k]


def get_polished_result(request_id: str) -> dict | None:
    """
    Return the stored LLM-enhanced answer for request_id, or None if not ready.
    Called by GET /api/query/result/{request_id}.
    """
    entry = _polish_store.get(request_id)
    if entry is None:
        return None
    if time.monotonic() - entry["ts"] > _STORE_TTL:
        del _polish_store[request_id]
        return None
    return entry

# Human-readable refusal messages
REFUSAL_MESSAGES = {
    RefusalReason.LOW_CONFIDENCE: "I couldn't find enough evidence in the knowledge base to answer that confidently.",
    RefusalReason.NO_EVIDENCE: "I couldn't find sufficient evidence in the knowledge base to answer that question.",
    RefusalReason.SAFETY_VIOLATION: "I can't help with that request.",
    RefusalReason.UNSAFE_INPUT: "I can't help with that request.",
    RefusalReason.PROMPT_INJECTION: "I can't help with that request.",
    RefusalReason.STT_FAILED: "I couldn't transcribe that recording. Please try again.",
    RefusalReason.GROUNDING_FAILED: "I couldn't verify the answer against the retrieved evidence. Here's what I found instead.",
    RefusalReason.GENERATION_UNAVAILABLE: "The answer engine is temporarily unavailable. Here is the strongest evidence I found.",
    RefusalReason.TIMEOUT: "That took longer than expected. Please try again.",
}


def _confidence_level(score: float, retrieval_score: float) -> ConfidenceLevel:
    if score >= 0.7 and retrieval_score >= 0.6:
        return ConfidenceLevel.HIGH
    if score >= 0.45:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.REFUSED


async def _background_polish(request_id: str, query: str, sources: list) -> None:
    """
    Background Groq generation. Result is stored in _polish_store so the
    frontend can poll /api/query/result/{request_id} and swap in the LLM answer.
    A Groq 429 or timeout is silently swallowed — the extractive answer already
    reached the user and is perfectly valid on its own.
    """
    try:
        gen = await generate(query, sources)
        if gen.status != "answered" or not gen.answer:
            return
        # Run full grounding on the generated answer before storing it
        grounding = verify_grounding(gen, sources)
        if not grounding.passed:
            logger.info(
                "Background polish grounding FAIL (score=%.3f) — not storing",
                grounding.score,
            )
            return
        # Opportunistic cleanup
        _sweep_store()
        _polish_store[request_id] = {
            "ts": time.monotonic(),
            "answer": gen.answer,
            "answer_source": "llm",
            "grounding_score": grounding.score,
            "cited_chunk_ids": gen.cited_chunk_ids,
        }
        logger.info(
            "Background polish stored: request_id=%s grounding=%.3f",
            request_id, grounding.score,
        )
    except Exception as exc:
        logger.debug("Background polish failed (non-blocking): %s", exc)


async def _background_safety_check(text: str, degraded_flag: list) -> None:
    """
    L2 Groq Prompt Guard check — runs concurrently with retrieval.
    L1 deterministic check already ran on the critical path. L2 is defense-in-depth.
    Groq free-tier takes ~500-800ms so it cannot block the <200ms SLA.
    If L2 fails/times-out, sets degraded_flag[0] = True so the response can
    surface safety_degraded=True to the frontend diagnostics panel.
    """
    try:
        await layer2_prompt_guard(text)
    except Exception:
        degraded_flag[0] = True  # L2 unavailable — L1 protection still applies


async def orchestrate_query(
    transcript_text: str,
    language: str = "auto",
) -> QueryResponse:
    """
    Full RAG pipeline from transcript text to structured answer.
    Returns the extractive fast-path answer in <200ms.
    Groq fires in the background (fire-and-forget, result discarded).
    """
    request_id = new_request_id()
    t_start = time.perf_counter()
    latency = LatencyMetrics()

    def elapsed_ms() -> float:
        return (time.perf_counter() - t_start) * 1000

    # ── Stage 1: Input provided from Edge ────────────────────────────────────
    transcript = transcript_text
    detected_lang = language
    latency.stt_ms = 0.0

    # ── Stage 2: Input normalization ─────────────────────────────────────────
    t_norm = time.perf_counter()
    normalized = normalize_text(transcript)
    latency.normalization_ms = (time.perf_counter() - t_norm) * 1000

    if not normalized:
        latency.full_e2e_ms = elapsed_ms()
        return QueryResponse(
            request_id=request_id,
            status=PipelineStatus.REFUSED,
            transcript=transcript,
            refusal_reason=RefusalReason.UNSAFE_INPUT,
            refusal_message="Empty or invalid query",
            latency=latency,
        )

    # ── Stage 3: Layer 1 guardrail (deterministic — free) ────────────────────
    l1 = layer1_check(normalized)
    if not l1.passed:
        latency.safety_ms = l1.latency_ms
        latency.full_e2e_ms = elapsed_ms()
        return QueryResponse(
            request_id=request_id,
            status=PipelineStatus.REFUSED,
            transcript=transcript,
            language=detected_lang,
            refusal_reason=l1.refusal_reason,
            refusal_message=REFUSAL_MESSAGES.get(l1.refusal_reason, l1.message or "Request not allowed"),
            latency=latency,
        )

    # ── Stage 4: Retrieval (fast path) + L2 Safety (background) ────────────────
    # L1 deterministic check already passed. L2 Groq safety runs as a background
    # task — it must NOT block the critical path.
    # Groq free-tier L2 takes ~500-800ms; awaiting it would dominate retrieval
    # (~15ms) and destroy the <200ms SLA. L2 is defense-in-depth; L1 already
    # covers injection/unsafe patterns deterministically.
    # degraded_flag[0] is set True by _background_safety_check if L2 fails.
    t_parallel = time.perf_counter()
    degraded_flag: list = [False]
    asyncio.create_task(_background_safety_check(normalized, degraded_flag))
    retrieval_result, embedding_ms = await hybrid_retrieve(normalized)

    parallel_ms = (time.perf_counter() - t_parallel) * 1000
    latency.safety_ms = l1.latency_ms  # only L1 cost counted on critical path
    latency.embedding_ms = embedding_ms
    latency.retrieval_ms = parallel_ms

    # ── Stage 5: Confidence gate ─────────────────────────────────────────────
    if not should_generate(retrieval_result):
        latency.rag_core_ms = elapsed_ms() - latency.stt_ms
        latency.full_e2e_ms = elapsed_ms()
        return QueryResponse(
            request_id=request_id,
            status=PipelineStatus.REFUSED,
            transcript=transcript,
            language=detected_lang,
            confidence=ConfidenceLevel.REFUSED,
            confidence_score=retrieval_result.confidence,
            sources=retrieval_result.sources[:3],
            refusal_reason=RefusalReason.LOW_CONFIDENCE,
            refusal_message=REFUSAL_MESSAGES[RefusalReason.LOW_CONFIDENCE],
            latency=latency,
        )

    sources = retrieval_result.sources

    # ── Stage 6: Extractive Fallback (Sub-millisecond) ────────────────────────
    t_extractive = time.perf_counter()
    fast_answer = extractive_fallback(sources, normalized, reason="fast_path")
    latency.generation_ms = (time.perf_counter() - t_extractive) * 1000
    
    if fast_answer.status == "refused":
        latency.rag_core_ms = elapsed_ms() - latency.stt_ms
        latency.full_e2e_ms = elapsed_ms()
        return QueryResponse(
            request_id=request_id,
            status=PipelineStatus.REFUSED,
            transcript=transcript,
            language=detected_lang,
            confidence=ConfidenceLevel.REFUSED,
            confidence_score=retrieval_result.confidence,
            sources=sources[:3],
            refusal_reason=fast_answer.refusal_reason,
            refusal_message=fast_answer.answer, # PASS THE DEBUG MESSAGE HERE
            latency=latency,
        )

    # ── Stage 7: Lightweight grounding (citation + entity overlap, ~0ms) ─────
    t_ground = time.perf_counter()
    grounding = verify_grounding_extractive(fast_answer, sources)
    latency.grounding_ms = (time.perf_counter() - t_ground) * 1000

    if not grounding.passed:
        # Grounding failed even for extractive — extremely rare but handle cleanly
        latency.rag_core_ms = elapsed_ms() - latency.stt_ms
        latency.full_e2e_ms = elapsed_ms()
        logger.warning(
            "Extractive grounding failed: score=%.3f — refusing with sources",
            grounding.score,
        )
        return QueryResponse(
            request_id=request_id,
            status=PipelineStatus.REFUSED,
            transcript=transcript,
            language=detected_lang,
            confidence=ConfidenceLevel.REFUSED,
            sources=sources[:3],  # always return evidence when we have it
            refusal_reason=RefusalReason.GROUNDING_FAILED,
            refusal_message=REFUSAL_MESSAGES[RefusalReason.GROUNDING_FAILED],
            grounding_score=grounding.score,
            latency=latency,
        )

    # ── Stage 8: Fire Groq in background (non-blocking) ─────────────────────
    # User gets their answer NOW from the fast extractive path.
    # Groq generates a polished version in the background and stores it in
    # _polish_store (keyed by request_id, TTL 60s). The frontend polls
    # /api/query/result/{request_id} ~1.5s later and swaps it in if available.
    # A 429 or timeout is silently swallowed — extractive answer is always valid.
    if settings.has_groq_key:
        asyncio.create_task(_background_polish(request_id, normalized, sources))

    # ── Final: build structured extractive response ───────────────────────────
    latency.rag_core_ms = elapsed_ms() - latency.stt_ms
    latency.full_e2e_ms = elapsed_ms()

    conf_level = _confidence_level(fast_answer.confidence, retrieval_result.confidence)

    logger.info(
        "Query complete: status=answered source=extractive confidence=%s rag_core=%.1fms",
        conf_level.value,
        latency.rag_core_ms,
        extra={"stage": "finalize", "latency_ms": round(latency.rag_core_ms, 1)},
    )

    return QueryResponse(
        request_id=request_id,
        status=PipelineStatus.ANSWERED,
        transcript=transcript,
        language=detected_lang,
        answer=fast_answer.answer,
        answer_source="extractive",
        confidence=conf_level,
        confidence_score=fast_answer.confidence,
        sources=sources[:5],
        refusal_reason=None,
        grounding_score=grounding.score,
        safety_degraded=degraded_flag[0],
        latency=latency,
    )
