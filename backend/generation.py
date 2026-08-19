"""
MangoVoice — Groq grounded generation.

Tool-contract approach: model must call either answer_from_context() or refuse().
Retried once on transient 5xx.
Max output: 128 tokens.
"""
from __future__ import annotations

import json
import time

from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

from backend.config import settings
from backend.schemas import GenerationResult, RetrievalSource, RefusalReason
from backend.telemetry import get_logger

logger = get_logger(__name__)


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are MangoVoice, a grounded RAG assistant. Your ONLY job is to answer questions using the evidence passages provided below.

STRICT RULES:
1. Answer ONLY from the provided evidence. Do not use any outside knowledge.
2. Every factual claim must be directly and explicitly stated in a cited evidence passage.
3. Keep answers concise (2-4 sentences maximum).
4. You MUST call exactly one of the two functions: answer_from_context or refuse.

WHEN TO CALL answer_from_context():
- At least one evidence passage directly and clearly answers the question.
- You can cite the specific passage(s) that contain the answer.
- The answer is factually present in the evidence — even if it is brief.
- If multiple passages together answer the question, use them all.

WHEN TO CALL refuse():
- NONE of the evidence passages directly answer the question asked.
- The evidence only mentions related keywords but in a completely different context (e.g., question is about blood type O+, evidence only discusses blood transfusion procedures with no mention of O positive being the most common type).
- You cannot form a factual answer without fabricating information not in the evidence.

IMPORTANT: If the evidence contains a clear, direct answer — even in just one passage — ALWAYS call answer_from_context(). 
Do NOT refuse just because the evidence is limited or the answer is short.
A brief but grounded answer is always better than a refusal when the evidence supports it.
"""

# ── Groq tool definitions (JSON Schema) ───────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "answer_from_context",
            "description": "Call this when the evidence is sufficient to answer the question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The grounded answer (2-4 sentences, based only on evidence)",
                    },
                    "cited_chunk_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "IDs of evidence chunks that support the answer",
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence 0.0-1.0 that the answer is fully grounded",
                    },
                },
                "required": ["answer", "cited_chunk_ids", "confidence"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "refuse",
            "description": "Call this when evidence is insufficient, irrelevant, or the question cannot be answered from the provided passages.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Brief reason why the question cannot be answered from evidence",
                    }
                },
                "required": ["reason"],
            },
        },
    },
]


def _build_evidence_block(sources: list[RetrievalSource]) -> str:
    lines = []
    for i, src in enumerate(sources[:settings.final_top_k]):
        lines.append(f"[{src.chunk_id}] ({src.language.upper()})\n{src.text.strip()}")
    return "\n\n---\n\n".join(lines)


@retry(
    stop=stop_after_attempt(2),
    wait=wait_fixed(0.3),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
async def _groq_call(messages: list[dict], strict: bool = False) -> dict:
    """Raw Groq API call with tool use."""
    from groq import AsyncGroq

    client = AsyncGroq(api_key=settings.groq_api_key)
    max_tokens = 256 if strict else max(384, settings.max_output_tokens)

    resp = await client.chat.completions.create(
        model=settings.groq_model,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        max_tokens=max_tokens,
        temperature=0.1,  # low temp for grounded generation
        timeout=settings.groq_timeout,
    )
    return resp


async def generate(
    query: str,
    sources: list[RetrievalSource],
    strict: bool = False,
) -> GenerationResult:
    """
    Generate a grounded answer via Groq.
    strict=True for regeneration with tighter constraints.
    """
    if not settings.has_groq_key:
        # Demo fallback when no key is configured
        return _demo_answer(sources)

    evidence_block = _build_evidence_block(sources)
    strict_note = "\n\nSTRICT REGENERATION: Each sentence must map directly to a cited evidence passage. Any sentence not clearly in the evidence must be removed." if strict else ""

    user_msg = f"""EVIDENCE:
{evidence_block}

QUESTION: {query}
{strict_note}

Call answer_from_context() with a complete 2-3 sentence answer that cites specific facts from the evidence above, or refuse() if the evidence does not answer the question."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    t0 = time.perf_counter()
    try:
        resp = await _groq_call(messages, strict=strict)
    except Exception as exc:
        logger.error("Groq generation failed: %s", exc, extra={"stage": "generation"})
        return GenerationResult(
            status="refused",
            refusal_reason=RefusalReason.GENERATION_UNAVAILABLE,
        )

    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "Groq generation complete",
        extra={"stage": "generation", "latency_ms": round(latency_ms, 1)},
    )

    choice = resp.choices[0]
    if not choice.message.tool_calls:
        return GenerationResult(
            status="refused",
            refusal_reason=RefusalReason.NO_EVIDENCE,
        )

    tool_call = choice.message.tool_calls[0]
    fn_name = tool_call.function.name
    try:
        args = json.loads(tool_call.function.arguments or "{}")
    except json.JSONDecodeError:
        args = {}

    if fn_name == "answer_from_context":
        answer = args.get("answer", "").strip()
        cited = args.get("cited_chunk_ids", [])
        confidence = float(args.get("confidence", 0.5))
        if not answer:
            return GenerationResult(status="refused", refusal_reason=RefusalReason.NO_EVIDENCE)
        return GenerationResult(
            status="answered",
            answer=answer,
            cited_chunk_ids=cited,
            confidence=confidence,
            regenerated=strict,
        )
    else:
        return GenerationResult(
            status="refused",
            refusal_reason=RefusalReason.NO_EVIDENCE,
        )


def _demo_answer(sources: list[RetrievalSource]) -> GenerationResult:
    """Demo fallback when GROQ_API_KEY is not set."""
    if not sources:
        return GenerationResult(status="refused", refusal_reason=RefusalReason.NO_EVIDENCE)
    top = sources[0]
    answer = f"Based on the retrieved evidence: {top.text[:200].strip()}..."
    return GenerationResult(
        status="answered",
        answer=answer,
        cited_chunk_ids=[top.chunk_id],
        confidence=0.72,
    )
