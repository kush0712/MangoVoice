"""
MangoVoice — Extractive fast-path answer.

Primary response path (Phase 1 redesign):
- Select the highest-confidence retrieved passage
- Extract the best matching sentence(s) via keyword overlap
- Return as the <200ms contractual answer

This is now the FIRST-CLASS answer, not a last resort.
Confidence is set to 0.65 (credible extractive evidence).
"""
from __future__ import annotations

import re

from backend.schemas import GenerationResult, RetrievalSource, RefusalReason


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\b[a-zA-Z\u0900-\u097F]{3,}\b", text.lower()))


def _best_sentence(passage: str, query: str) -> str:
    """Extract the sentence from a passage that best matches the query."""
    sentences = re.split(r"(?<=[.!?])\s+", passage.strip())
    query_tokens = _tokenize(query)
    if not sentences:
        return passage[:300]

    scored = [
        (len(_tokenize(s) & query_tokens), s)
        for s in sentences
        if len(s.strip()) > 15
    ]
    if not scored:
        return sentences[0][:300]

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def extractive_fallback(
    sources: list[RetrievalSource],
    query: str,
    reason: str = "fast_path",
) -> GenerationResult:
    """
    Return the best extractive snippet from top sources.

    Primary fast path: called before any LLM invocation so the system
    can return a grounded, verifiable answer in <200ms.
    """
    if not sources:
        return GenerationResult(
            status="refused",
            refusal_reason=RefusalReason.NO_EVIDENCE,
        )

    top = sources[0]
    snippet = _best_sentence(top.text, query)

    answer = (
        f'Based on the retrieved evidence:\n\n'
        f'"{snippet}"\n\n'
        f"Source: {top.chunk_id}"
    )

    return GenerationResult(
        status="answered",
        answer=answer,
        cited_chunk_ids=[top.chunk_id],
        confidence=0.65,  # credible extractive evidence — this is the primary response
        refusal_reason=None,
    )
