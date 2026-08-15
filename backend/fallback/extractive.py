"""
MangoVoice — Extractive fallback.

When Groq is unavailable or grounding fails twice:
- Select the highest-confidence retrieved passage
- Extract the best matching sentence(s) via keyword overlap
- Return as an evidence-only answer

This keeps the system useful even when generation fails.
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
    reason: str = "generation_unavailable",
) -> GenerationResult:
    """
    Return the best extractive snippet from top sources.
    Used as a last resort when generation / grounding repeatedly fails.
    """
    if not sources:
        return GenerationResult(
            status="refused",
            refusal_reason=RefusalReason.NO_EVIDENCE,
        )

    top = sources[0]
    snippet = _best_sentence(top.text, query)

    answer = (
        f"I couldn't generate a full answer right now.\n\n"
        f"The strongest evidence I found is:\n\n"
        f'"{snippet}"\n\n'
        f"Source: {top.chunk_id} (score: {top.score:.2f})"
    )

    return GenerationResult(
        status="answered",
        answer=answer,
        cited_chunk_ids=[top.chunk_id],
        confidence=0.35,  # low confidence — clearly extractive
        refusal_reason=None,
    )
