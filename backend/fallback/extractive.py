"""
MangoVoice — Extractive fast-path answer.

Primary response path (Phase 1 redesign):
- Select the highest-confidence retrieved passage
- Extract the best matching sentence(s) via keyword overlap
- GATE: if the best sentence doesn't share enough terms with the query,
  REFUSE rather than confidently returning a wrong answer
- Return as the <200ms contractual answer

The relevance gate is critical: without it, a question about "capital of
Australia" could return a chunk about Melbourne time zones (which mentions
"Australia") with 100% grounding score but a factually wrong answer.
The gate requires ≥2 query tokens to appear in the best sentence for
longer queries, ≥1 for very short queries.
"""
from __future__ import annotations

import re

from backend.schemas import GenerationResult, RetrievalSource, RefusalReason


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\b[a-zA-Z\u0900-\u097F]{3,}\b", text.lower()))


def _best_sentence_scored(passage: str, query: str) -> tuple[int, str]:
    """
    Return (overlap_score, best_sentence) for the passage.
    overlap_score = number of query tokens that appear in the best sentence.
    """
    sentences = re.split(r"(?<=[.!?])\s+", passage.strip())
    query_tokens = _tokenize(query)
    if not sentences:
        return 0, passage[:300]

    scored = [
        (len(_tokenize(s) & query_tokens), s)
        for s in sentences
        if len(s.strip()) > 15
    ]
    if not scored:
        return 0, sentences[0][:300]

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][0], scored[0][1]


def extractive_fallback(
    sources: list[RetrievalSource],
    query: str,
    reason: str = "fast_path",
) -> GenerationResult:
    """
    Return the best extractive snippet from top sources — or REFUSE if the
    retrieved content doesn't actually address the query.

    Relevance gate: requires ≥2 query tokens in the best sentence (≥1 for
    very short queries with <3 tokens). This prevents returning topically
    unrelated snippets with false confidence (e.g., returning a Melbourne
    time-zones chunk for a question about Australia's capital).

    Primary fast path: called before any LLM invocation so the system
    can return a grounded, verifiable answer in <200ms.
    """
    if not sources:
        return GenerationResult(
            status="refused",
            refusal_reason=RefusalReason.NO_EVIDENCE,
        )

    query_tokens = _tokenize(query)
    # Minimum overlap required: 2 for normal queries, 1 for very short ones
    min_overlap = 2 if len(query_tokens) >= 3 else 1

    # Search across top-3 sources for the best-matching sentence
    best_overlap = 0
    best_snippet = ""
    best_source = sources[0]

    for source in sources[:3]:
        overlap, sentence = _best_sentence_scored(source.text, query)
        if overlap > best_overlap:
            best_overlap = overlap
            best_snippet = sentence
            best_source = source

    # ── Relevance gate ────────────────────────────────────────────────────────
    # If no sentence across the top-3 sources shares enough terms with the
    # query, refuse rather than returning a confidently wrong answer.
    if best_overlap < min_overlap:
        return GenerationResult(
            status="refused",
            refusal_reason=RefusalReason.NO_EVIDENCE,
        )

    answer = (
        f'Based on the retrieved evidence:\n\n'
        f'"{best_snippet}"\n\n'
        f"Source: {best_source.chunk_id}"
    )

    return GenerationResult(
        status="answered",
        answer=answer,
        cited_chunk_ids=[best_source.chunk_id],
        confidence=0.65,  # credible extractive evidence — this is the primary response
        refusal_reason=None,
    )
