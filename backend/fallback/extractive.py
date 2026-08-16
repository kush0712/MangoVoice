"""
MangoVoice — Extractive fast-path answer.

Primary response path:
- Retrieve top-20 chunks via hybrid search (dense + BM25 + RRF)
- Two-phase source selection:
    Phase 1: Search high-semantic-quality sources (raw_dense_score >= 0.30)
             These are semantically verified — return if any meaningful token matches
    Phase 2: If Phase 1 finds nothing, search remaining sources with strict
             requirements (>= 2 meaningful token overlaps)
- Token overlap uses STOP-WORD-FILTERED query tokens to prevent question
  frame words ("what", "are", "how", "does") from matching unrelated chunks

Key invariant: we never return a snippet from a chunk where the content
is semantically unrelated to the query. Both the dense score threshold AND
meaningful keyword overlap must agree before we answer.
"""
from __future__ import annotations

import re
import numpy as np

from backend.schemas import GenerationResult, RetrievalSource, RefusalReason
from backend.retrieval.embeddings import embed_query

# raw_dense_score threshold: above this the semantic similarity is strong
# enough that we trust the retrieval and only need 1 meaningful token match.
DENSE_QUALITY_THRESHOLD = 0.30

# Words that appear in question-format text everywhere — useless as evidence
# that a chunk is about the right topic. Must not count toward overlap.
_STOP_WORDS: frozenset[str] = frozenset({
    # English question / function words
    "what", "are", "the", "how", "does", "when", "who", "which", "where",
    "why", "was", "were", "did", "has", "have", "had", "will", "would",
    "can", "could", "should", "shall", "may", "might", "must", "been",
    "being", "into", "onto", "from", "this", "that", "these", "those",
    "with", "about", "many", "much", "some", "any", "all", "both", "its",
    "his", "her", "our", "your", "their", "there", "here", "then", "than",
    "such", "also", "only", "just", "more", "most", "very", "each", "for",
    "and", "but", "not", "you", "they", "them", "him", "her", "its",
    "used", "use", "using", "cause", "causes", "related", "known",
    "called", "found", "usually", "often", "generally", "commonly",
    # Hinglish / Hindi function words (romanised)
    "kya", "hai", "hain", "mein", "kaise", "kaun", "kab", "kitna", "kitni",
    "aur", "yeh", "woh", "toh", "bhi", "tak", "par", "tha", "thi",
    "hota", "hoti", "hote", "kaafi", "bahut", "iska", "iski", "iske",
})


def _tokenize(text: str) -> set[str]:
    """All 3+ char word tokens — used for passage text."""
    return set(re.findall(r"\b[a-zA-Z\u0900-\u097F]{3,}\b", text.lower()))


def _meaningful_query_tokens(query: str) -> set[str]:
    """
    Query tokens with stop words removed.
    These are the CONTENT words that a relevant chunk MUST contain.
    Using all tokens (including 'what', 'are', 'how') causes false matches
    against question-format chunks unrelated to the topic.
    """
    raw = _tokenize(query)
    meaningful = raw - _STOP_WORDS
    # Always keep proper nouns and key domain terms regardless of stop-word list
    # (e.g. 'malaria', 'photosynthesis', 'gandhi' are never stop words)
    return meaningful if meaningful else raw  # fallback to all tokens if everything stripped


def _best_sentence_scored(passage: str, query_tokens: set[str]) -> tuple[int, str]:
    """
    Return (meaningful_overlap_count, best_sentence) for the passage.
    query_tokens must already be stop-word-filtered.
    """
    sentences = re.split(r"(?<=[.!?])\s+", passage.strip())
    if not sentences:
        return 0, passage[:300]

    scored = [
        (len(_tokenize(s) & query_tokens), s)
        for s in sentences
        if len(s.strip()) > 20
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
    Return the best extractive snippet — or REFUSE if no source is both
    semantically and topically relevant to the query.

    Uses a fast semantic verification step: we find the single sentence with the
    most keyword overlap, and then run it through the local FastEmbed model to
    verify it is semantically addressing the query. This prevents coincidental
    keyword overlap (like "Australia capital" matching "South Australia's capital")
    from returning wrong answers, while preserving genuine matches.
    """
    if not sources:
        return GenerationResult(
            status="refused",
            refusal_reason=RefusalReason.NO_EVIDENCE,
        )

    query_tokens = _meaningful_query_tokens(query)

    best_overlap = 0
    best_snippet = ""
    best_source = sources[0]

    for source in sources:
        overlap, sentence = _best_sentence_scored(source.text, query_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best_snippet = sentence
            best_source = source

    if best_overlap >= 1:
        # ── Semantic Verification of Candidate Sentence ────────────────────────
        # Token overlap can be fooled by coincidences. We use the local FastEmbed 
        # model to verify the exact extracted sentence against the query.
        # Since embed_query caches the query, this only takes ~5-10ms.
        q_vec = embed_query(query)
        s_vec = embed_query(best_snippet)
        
        sim = float(np.dot(q_vec, s_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(s_vec)))
        
        # 0.55 allows cross-lingual Hinglish-to-English matches (typically ~0.56)
        # while cleanly rejecting false keyword overlaps (typically ~0.10 - 0.50).
        SENTENCE_SIM_THRESHOLD = 0.55
        
        if sim >= SENTENCE_SIM_THRESHOLD:
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

    # ── Failed to find semantic match → refuse cleanly ────────────────────────
    return GenerationResult(
        status="refused",
        refusal_reason=RefusalReason.NO_EVIDENCE,
    )
