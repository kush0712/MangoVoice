"""
MangoVoice — Extractive fast-path answer.

Primary response path:
- Retrieve top-8 chunks via hybrid search
- Search ALL 8 sources for the best query-matching sentence
- ADAPTIVE relevance gate:
    • raw_dense_score >= 0.30 (good semantic retrieval) → require ≥1 token overlap
    • raw_dense_score  < 0.30 (marginal retrieval)      → require ≥2 token overlap
    • overlap == 0 → always refuse
- Return as the <200ms contractual answer

Calibrated against real retrieval scores:
  Gandhi Hindi:     dense=0.374  → high quality, 1 overlap sufficient
  Blood pressure:   dense=0.523  → high quality, 1 overlap sufficient
  Diabetes:         dense=0.354  → high quality, 1 overlap sufficient
  Australia capital: dense=0.145 → marginal, needs 2 overlaps (capital absent → refuses)

This maximises recall when the dataset has the answer while preventing
confidently wrong answers when retrieval is weak.
"""
from __future__ import annotations

import re

from backend.schemas import GenerationResult, RetrievalSource, RefusalReason

# raw_dense_score threshold: above this the retrieval is semantically strong
# and we trust it with just 1 token overlap.
# Below this we need 2 token overlaps to guard against weak/coincidental matches.
DENSE_QUALITY_THRESHOLD = 0.30


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\b[a-zA-Z\u0900-\u097F]{3,}\b", text.lower()))


def _best_sentence_scored(passage: str, query_tokens: set[str]) -> tuple[int, str]:
    """
    Return (overlap_score, best_sentence) for the passage.
    overlap_score = number of query tokens that appear in the best sentence.
    query_tokens is pre-computed and passed in to avoid re-tokenizing per source.
    """
    sentences = re.split(r"(?<=[.!?])\s+", passage.strip())
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
    Return the best extractive snippet from ALL retrieved sources — or REFUSE
    if no source has a snippet that's relevant to the query.

    Adaptive relevance gate:
    - Searches ALL sources (not just top-3) to maximise recall
    - raw_dense_score >= 0.30 → good semantic retrieval → require ≥1 token overlap
    - raw_dense_score  < 0.30 → marginal retrieval     → require ≥2 token overlap
    - overlap == 0 → always refuse (no query terms found anywhere)

    This ensures: if the dataset has the answer and retrieval ranked it
    above confidence threshold, we will answer. We only refuse when either
    (a) the content genuinely doesn't address the query, or (b) retrieval
    quality is low AND no strong keyword match exists.
    """
    if not sources:
        return GenerationResult(
            status="refused",
            refusal_reason=RefusalReason.NO_EVIDENCE,
        )

    query_tokens = _tokenize(query)

    # ── Search ALL sources for the best query-matching sentence ──────────────
    # Searching only top-3 risks missing the correct chunk at position 4-8.
    best_overlap = 0
    best_snippet = ""
    best_source = sources[0]

    for source in sources:
        overlap, sentence = _best_sentence_scored(source.text, query_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best_snippet = sentence
            best_source = source

    # ── Adaptive relevance gate ───────────────────────────────────────────────
    if best_overlap == 0:
        # Zero query terms found in any retrieved chunk — unambiguously refuse
        return GenerationResult(
            status="refused",
            refusal_reason=RefusalReason.NO_EVIDENCE,
        )

    # Minimum overlap required depends on how semantically strong the retrieval is
    if best_source.raw_dense_score >= DENSE_QUALITY_THRESHOLD:
        # Good retrieval quality: cosine similarity confirms semantic relevance.
        # One overlapping token is sufficient confirmation.
        min_overlap = 1
    else:
        # Marginal retrieval: cosine similarity is weak (coincidental match like
        # "Australia" appearing in a time-zones chunk). Require 2+ tokens to
        # confirm the snippet actually addresses the question.
        min_overlap = 2 if len(query_tokens) >= 3 else 1

    if best_overlap < min_overlap:
        return GenerationResult(
            status="refused",
            refusal_reason=RefusalReason.NO_EVIDENCE,
        )

    # ── Return the grounded extractive answer ─────────────────────────────────
    answer = (
        f'Based on the retrieved evidence:\n\n'
        f'"{best_snippet}"\n\n'
        f"Source: {best_source.chunk_id}"
    )

    return GenerationResult(
        status="answered",
        answer=answer,
        cited_chunk_ids=[best_source.chunk_id],
        confidence=0.65,
        refusal_reason=None,
    )
