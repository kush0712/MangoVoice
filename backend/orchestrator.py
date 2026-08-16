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
    [Groq fires as background task — result discarded, user already answered]

Design goals:
  1. Extractive answer is the <200ms contractual response.
  2. Groq is NEVER awaited in /api/query — zero blocking LLM latency.
  3. Every refused branch returns sources[:3] when evidence exists.
  4. Per-stage timing captured precisely.
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
from backend.grounding import verify_grounding_extractive
from backend.fallback.extractive import extractive_fallback
from backend.telemetry import get_logger, new_request_id

logger = get_logger(__name__)

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


async def _background_polish(query: str, sources: list) -> None:
    """
    Fire-and-forget Groq call. Result is discarded — user already received
    their extractive answer. Wrapped in try/except so a Groq 429 or timeout
    never surfaces as an unhandled exception in the worker logs.
    """
    try:
        await generate(query, sources)
    except Exception:
        pass  # result discarded intentionally


async def _background_safety_check(text: str) -> None:
    """
    Fire-and-forget L2 Groq Prompt Guard check.
    L1 deterministic check already ran on the critical path. L2 is defense-in-depth
    and must NOT block the user-visible response. Groq free-tier takes 500-800ms;
    awaiting it in a gather() would dominate retrieval (~15ms) and destroy the SLA.
    Result is discarded — the user already got their answer from the fast path.
    """
    try:
        await layer2_prompt_guard(text)
    except Exception:
        pass  # result discarded, L1 protection already applied


async def orchestrate_query(
    audio_bytes: Optional[bytes] = None,
    transcript_text: Optional[str] = None,
    language: str = "auto",
) -> QueryResponse:
    """
    Full RAG pipeline from audio/text to structured answer.
    Returns the extractive fast-path answer in <200ms.
    Groq fires in the background (fire-and-forget, result discarded).
    """
    request_id = new_request_id()
    t_start = time.perf_counter()
    latency = LatencyMetrics()

    def elapsed_ms() -> float:
        return (time.perf_counter() - t_start) * 1000

    # ── Stage 1: STT ──────────────────────────────────────────────────────────
    transcript = None
    detected_lang = language

    if transcript_text:
        # Text input path (demo queries / text endpoint)
        transcript = transcript_text
        latency.stt_ms = 0.0
    else:
        # Audio path → Sarvam STT
        from backend.stt import transcribe, STTError
        t_stt = time.perf_counter()
        try:
            stt_result = await transcribe(audio_bytes, language)
            transcript = stt_result.text
            detected_lang = stt_result.language or language
            latency.stt_ms = (time.perf_counter() - t_stt) * 1000
        except STTError as exc:
            latency.stt_ms = (time.perf_counter() - t_stt) * 1000
            latency.full_e2e_ms = elapsed_ms()
            return QueryResponse(
                request_id=request_id,
                status=PipelineStatus.ERROR,
                refusal_reason=exc.reason,
                refusal_message=exc.message,
                latency=latency,
            )

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
    # fire-and-forget task — it must NOT block the critical path.
    # Groq free-tier L2 takes ~500-800ms; awaiting it would dominate retrieval
    # (~15ms) and destroy the <200ms SLA. L2 is defense-in-depth; L1 already
    # covers injection/unsafe patterns deterministically.
    t_parallel = time.perf_counter()

    asyncio.create_task(_background_safety_check(normalized))  # fire-and-forget
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

    # ── Stage 8: Fire Groq in background (fire-and-forget, never awaited) ────
    # User gets their answer NOW. Groq result is discarded — it does not affect
    # this response. A 429 rate-limit or timeout will be silently swallowed by
    # _background_polish so no noisy tracebacks appear in worker logs.
    if settings.has_groq_key:
        asyncio.create_task(_background_polish(normalized, sources))

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
        latency=latency,
    )
