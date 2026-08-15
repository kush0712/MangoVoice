"""
MangoVoice — Async RAG orchestrator (state machine).

States:
  RECEIVED → TRANSCRIBING → TRANSCRIBED
  → SAFETY_CHECK + RETRIEVAL (parallel)
  → CONFIDENCE_GATE
  │ REFUSE
  └ GENERATE → GROUNDING_CHECK
                │ PASS → FINALIZE
                └ REGENERATE_ONCE → FINALIZE

Per-stage timing is captured precisely.
All retries are bounded (max 1 retry per external call).
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
from backend.grounding import verify_grounding
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


async def orchestrate_query(
    audio_bytes: Optional[bytes] = None,
    transcript_text: Optional[str] = None,
    language: str = "auto",
) -> QueryResponse:
    """
    Full RAG pipeline from audio/text to structured answer.
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

    # ── Stage 4: Safety (Layer 2) + Retrieval in PARALLEL ────────────────────
    t_parallel = time.perf_counter()

    safety_task = asyncio.create_task(layer2_prompt_guard(normalized))
    retrieval_task = asyncio.create_task(hybrid_retrieve(normalized))

    (safety_result, (retrieval_result, embedding_ms)) = await asyncio.gather(
        safety_task, retrieval_task
    )
    parallel_ms = (time.perf_counter() - t_parallel) * 1000

    latency.safety_ms = max(safety_result.latency_ms, 1.0)
    latency.embedding_ms = embedding_ms
    latency.retrieval_ms = parallel_ms - embedding_ms  # approximate

    # Safety check result
    if not safety_result.passed:
        latency.full_e2e_ms = elapsed_ms()
        latency.rag_core_ms = elapsed_ms() - latency.stt_ms
        return QueryResponse(
            request_id=request_id,
            status=PipelineStatus.REFUSED,
            transcript=transcript,
            language=detected_lang,
            refusal_reason=safety_result.refusal_reason,
            refusal_message=REFUSAL_MESSAGES.get(
                safety_result.refusal_reason, "I can't help with that request."
            ),
            latency=latency,
        )

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

    # ── Stage 6: Generation ──────────────────────────────────────────────────
    t_gen = time.perf_counter()
    gen_result: GenerationResult = await generate(normalized, sources)
    latency.generation_ms = (time.perf_counter() - t_gen) * 1000

    # Handle generation failure → clean refusal (do NOT dump unrelated extractive chunks)
    if gen_result.status == "refused" and gen_result.refusal_reason == RefusalReason.GENERATION_UNAVAILABLE:
        latency.rag_core_ms = elapsed_ms() - latency.stt_ms
        latency.full_e2e_ms = elapsed_ms()
        return QueryResponse(
            request_id=request_id,
            status=PipelineStatus.REFUSED,
            transcript=transcript,
            language=detected_lang,
            confidence=ConfidenceLevel.REFUSED,
            sources=[],
            refusal_reason=RefusalReason.GENERATION_UNAVAILABLE,
            refusal_message=REFUSAL_MESSAGES[RefusalReason.GENERATION_UNAVAILABLE],
            latency=latency,
        )

    # Model chose to refuse
    if gen_result.status == "refused":
        latency.rag_core_ms = elapsed_ms() - latency.stt_ms
        latency.full_e2e_ms = elapsed_ms()
        return QueryResponse(
            request_id=request_id,
            status=PipelineStatus.REFUSED,
            transcript=transcript,
            language=detected_lang,
            confidence=ConfidenceLevel.REFUSED,
            sources=sources[:3],
            refusal_reason=RefusalReason.NO_EVIDENCE,
            refusal_message=REFUSAL_MESSAGES[RefusalReason.NO_EVIDENCE],
            latency=latency,
        )

    # ── Stage 7: Grounding verification ─────────────────────────────────────
    t_ground = time.perf_counter()
    grounding = verify_grounding(gen_result, sources)
    latency.grounding_ms = (time.perf_counter() - t_ground) * 1000

    # If grounding fails → one strict regeneration
    if not grounding.passed:
        logger.info("Grounding failed — attempting strict regeneration")
        t_regen = time.perf_counter()
        gen_result = await generate(normalized, sources, strict=True)
        latency.generation_ms += (time.perf_counter() - t_regen) * 1000

        if gen_result.status == "answered":
            t_ground2 = time.perf_counter()
            grounding = verify_grounding(gen_result, sources)
            latency.grounding_ms += (time.perf_counter() - t_ground2) * 1000

        if not grounding.passed or gen_result.status != "answered":
            # Grounding failed twice → refuse instead of dumping unrelated text
            latency.rag_core_ms = elapsed_ms() - latency.stt_ms
            latency.full_e2e_ms = elapsed_ms()
            return QueryResponse(
                request_id=request_id,
                status=PipelineStatus.REFUSED,
                transcript=transcript,
                language=detected_lang,
                confidence=ConfidenceLevel.REFUSED,
                sources=[],
                refusal_reason=RefusalReason.GROUNDING_FAILED,
                refusal_message="I couldn't find evidence in the knowledge base that answers that question.",
                grounding_score=grounding.score,
                latency=latency,
            )

    # ── Final: build structured response ─────────────────────────────────────
    latency.rag_core_ms = elapsed_ms() - latency.stt_ms
    latency.full_e2e_ms = elapsed_ms()

    conf_level = _confidence_level(gen_result.confidence, retrieval_result.confidence)

    logger.info(
        "Query complete: status=answered confidence=%s rag_core=%.1fms",
        conf_level.value,
        latency.rag_core_ms,
        extra={"stage": "finalize", "latency_ms": round(latency.rag_core_ms, 1)},
    )

    # Filter sources to only cited ones (+ top 3 for context)
    cited_ids = set(gen_result.cited_chunk_ids)
    displayed_sources = [s for s in sources if s.chunk_id in cited_ids]
    if len(displayed_sources) < 3:
        displayed_sources += [s for s in sources if s.chunk_id not in cited_ids][
            : 3 - len(displayed_sources)
        ]

    return QueryResponse(
        request_id=request_id,
        status=PipelineStatus.ANSWERED,
        transcript=transcript,
        language=detected_lang,
        answer=gen_result.answer,
        confidence=conf_level,
        confidence_score=gen_result.confidence,
        sources=displayed_sources[:5],
        refusal_reason=None,
        grounding_score=grounding.score,
        latency=latency,
    )
