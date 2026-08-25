"""
MangoVoice — Eval Loop Generator Module.
Conforms to TARGET_INTERFACE.md for rag-local-eval-loop.
"""
from __future__ import annotations

import os
import sys
import re
from typing import Any, List, Optional
import numpy as np

# Ensure repository root is on sys.path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from backend.retrieval.embeddings import get_embedder

# Stop-words to prevent question-frame words from creating false matches
_STOP_WORDS: frozenset[str] = frozenset({
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
    "kya", "hai", "hain", "mein", "kaise", "kaun", "kab", "kitna", "kitni",
    "aur", "yeh", "woh", "toh", "bhi", "tak", "par", "tha", "thi",
    "hota", "hoti", "hote", "kaafi", "bahut", "iska", "iski", "iske",
})


class Answer:
    """Answer object meeting rag-local-eval-loop TARGET_INTERFACE contract."""

    def __init__(
        self,
        text: str,
        grounded: bool,
        confidence: float = 1.0,
        cited_source: Optional[str] = None,
    ):
        self.text = text
        self.grounded = bool(grounded)
        self.confidence = float(confidence)
        self.cited_source = cited_source

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return f"Answer(text={self.text!r}, grounded={self.grounded})"


def _tokenize(text: str) -> set[str]:
    """Extract word tokens (3+ chars) from text."""
    return set(re.findall(r"\b[a-zA-Z\u0900-\u097F]{3,}\b", text.lower()))


def _meaningful_query_tokens(query: str) -> set[str]:
    """Filter out common question frame words to isolate key topic terms."""
    raw = _tokenize(query)
    meaningful = raw - _STOP_WORDS
    return meaningful if meaningful else raw


def _extract_text_and_source(item: Any) -> tuple[str, str]:
    """Extract text and source identifier from diverse input object shapes."""
    if hasattr(item, "text"):
        text = str(getattr(item, "text", ""))
    elif isinstance(item, dict):
        text = str(item.get("text") or item.get("passage") or "")
    else:
        text = str(item)

    if hasattr(item, "source"):
        source = str(getattr(item, "source", ""))
    elif isinstance(item, dict):
        source = str(item.get("source") or item.get("id") or "")
    else:
        source = "evidence"

    return text.strip(), source


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def generate_answer(query: str, results: list) -> Answer:
    """
    Generate an answer from retrieved candidate passages.

    Fulfills rag-local-eval-loop contract:
    - query: str
    - results: list of candidate objects with .text and .source attributes
    - returns: Answer instance with .text: str and .grounded: bool
    """
    if not results or not query or not query.strip():
        return Answer(
            text="I cannot answer this question based on the provided evidence.",
            grounded=False,
            confidence=0.0,
        )

    # 1. Normalize query and candidate passages
    clean_query = query.strip()
    query_tokens = _meaningful_query_tokens(clean_query)
    embedder = get_embedder()
    query_vec = embedder.embed_one(clean_query)

    best_score = -1.0
    best_sentence = ""
    best_source = ""

    for item in results:
        text, source = _extract_text_and_source(item)
        if not text or len(text) < 15:
            continue

        # Split passage into distinct sentences
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 15]
        if not sentences:
            sentences = [text[:350].strip()]

        for sent in sentences:
            sent_tokens = _tokenize(sent)
            overlap = len(sent_tokens & query_tokens)
            
            # Fast filter: skip sentences with zero keyword overlap unless candidate text is short
            if overlap == 0 and len(sentences) > 1:
                continue

            sent_vec = embedder.embed_one(sent)
            sim = _cosine_sim(query_vec, sent_vec)

            # Combined heuristic: semantic similarity + keyword overlap boost
            composite_score = sim + (0.05 * min(overlap, 3))

            if composite_score > best_score:
                best_score = composite_score
                best_sentence = sent
                best_source = source

    # Relevance verification:
    # High confidence (sim >= 0.45 or strong overlap >= 2 with sim >= 0.32)
    # Refuses unanswerable distractor candidate passages
    is_grounded = (best_score >= 0.45 and bool(best_sentence)) or (
        best_score >= 0.35 and len(_tokenize(best_sentence) & query_tokens) >= 2
    )

    if is_grounded and best_sentence:
        # Return the grounded factual answer
        return Answer(
            text=best_sentence,
            grounded=True,
            confidence=min(1.0, float(best_score)),
            cited_source=best_source,
        )
    else:
        # Grounded refusal on unanswerable queries
        return Answer(
            text="I cannot answer this question based on the provided evidence.",
            grounded=False,
            confidence=0.0,
        )
