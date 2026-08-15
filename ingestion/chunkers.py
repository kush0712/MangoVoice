"""
MangoVoice — 5 Chunking Strategies.

Strategy A: Canonical passage (original MSMARCO passage as-is)
Strategy B: Sentence windows (2-sent, 1-sentence overlap)
Strategy C: Fixed token windows (128 tokens, 32 overlap)
Strategy D: Semantic boundary splitting (cosine similarity breakpoints)
Strategy E: Parent-child hierarchical (parent ~350 tok, child ~100 tok)

Each strategy returns: List[Dict] with keys:
  chunk_id, parent_id, query_id, language, strategy,
  chunk_start, chunk_end, text
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Generator


@dataclass
class Chunk:
    chunk_id: str
    parent_id: str
    query_id: str
    language: str
    strategy: str
    chunk_start: int
    chunk_end: int
    text: str
    vector: list[float] = field(default_factory=list)


# ── Utilities ─────────────────────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using regex."""
    # Handle both ASCII punctuation and Devanagari danda (।)
    sentences = re.split(r"(?<=[.!?।])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _tokenize_simple(text: str) -> list[str]:
    """Word-level tokenization (no NLTK required)."""
    return re.findall(r"\S+", text)


def _chunk_tokens(tokens: list[str], size: int, overlap: int) -> list[list[str]]:
    """Sliding window over a token list."""
    if len(tokens) <= size:
        return [tokens]
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + size
        chunks.append(tokens[start:end])
        start += size - overlap
    return chunks


def _make_id() -> str:
    return uuid.uuid4().hex[:12]


# ── Strategy A — Canonical passage ───────────────────────────────────────────

def strategy_a_canonical(
    passage: str,
    parent_id: str,
    query_id: str,
    language: str,
) -> list[Chunk]:
    """Use the original passage as-is. 1 chunk per passage."""
    if not passage.strip():
        return []
    return [
        Chunk(
            chunk_id=f"A-{parent_id}",
            parent_id=parent_id,
            query_id=query_id,
            language=language,
            strategy="canonical",
            chunk_start=0,
            chunk_end=len(passage),
            text=passage.strip(),
        )
    ]


# ── Strategy B — Sentence windows ────────────────────────────────────────────

def strategy_b_sentence_windows(
    passage: str,
    parent_id: str,
    query_id: str,
    language: str,
    window: int = 2,
    overlap: int = 1,
) -> list[Chunk]:
    """2-sentence windows with 1-sentence overlap."""
    sentences = _split_sentences(passage)
    if not sentences:
        return []

    chunks = []
    step = max(1, window - overlap)
    for i in range(0, len(sentences), step):
        window_sents = sentences[i : i + window]
        text = " ".join(window_sents)
        if len(text) < 10:
            continue
        chunks.append(
            Chunk(
                chunk_id=f"B-{parent_id}-{i}",
                parent_id=parent_id,
                query_id=query_id,
                language=language,
                strategy="sentence_window",
                chunk_start=i,
                chunk_end=i + len(window_sents),
                text=text,
            )
        )
    return chunks


# ── Strategy C — Fixed token windows ─────────────────────────────────────────

def strategy_c_fixed_token(
    passage: str,
    parent_id: str,
    query_id: str,
    language: str,
    token_size: int = 128,
    overlap: int = 32,
) -> list[Chunk]:
    """Fixed 128-token windows with 32-token overlap."""
    tokens = _tokenize_simple(passage)
    if not tokens:
        return []

    windows = _chunk_tokens(tokens, token_size, overlap)
    chunks = []
    char_pos = 0
    for i, window_tokens in enumerate(windows):
        text = " ".join(window_tokens)
        if len(text) < 10:
            continue
        chunks.append(
            Chunk(
                chunk_id=f"C-{parent_id}-{i}",
                parent_id=parent_id,
                query_id=query_id,
                language=language,
                strategy="fixed_token",
                chunk_start=char_pos,
                chunk_end=char_pos + len(text),
                text=text,
            )
        )
        char_pos += len(text)
    return chunks


# ── Strategy D — Semantic boundary splitting ──────────────────────────────────

def strategy_d_semantic(
    passage: str,
    parent_id: str,
    query_id: str,
    language: str,
    embedder=None,
    similarity_threshold: float = 0.75,
    min_chunk_tokens: int = 30,
    max_chunk_tokens: int = 200,
) -> list[Chunk]:
    """
    Split at semantic boundaries where adjacent sentence similarity drops.
    Requires embedder. Falls back to sentence windows if embedder unavailable.
    """
    sentences = _split_sentences(passage)
    if len(sentences) <= 2:
        return strategy_b_sentence_windows(passage, parent_id, query_id, language)

    if embedder is None:
        return strategy_b_sentence_windows(passage, parent_id, query_id, language)

    import numpy as np

    try:
        vecs = embedder.embed(sentences)  # (N, 384)
    except Exception:
        return strategy_b_sentence_windows(passage, parent_id, query_id, language)

    # Compute cosine similarity between adjacent sentences
    def cosine(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0

    similarities = [cosine(vecs[i], vecs[i + 1]) for i in range(len(vecs) - 1)]

    # Find breakpoints where similarity drops below threshold
    breakpoints = {0}
    for i, sim in enumerate(similarities):
        if sim < similarity_threshold:
            breakpoints.add(i + 1)
    breakpoints.add(len(sentences))

    sorted_bp = sorted(breakpoints)
    groups = [sentences[sorted_bp[i] : sorted_bp[i + 1]] for i in range(len(sorted_bp) - 1)]

    # Merge tiny groups and enforce max size
    merged: list[list[str]] = []
    for g in groups:
        token_count = len(_tokenize_simple(" ".join(g)))
        if token_count < min_chunk_tokens and merged:
            merged[-1].extend(g)
        elif token_count > max_chunk_tokens:
            # Split large group at midpoint
            mid = len(g) // 2
            merged.append(g[:mid])
            merged.append(g[mid:])
        else:
            merged.append(g)

    chunks = []
    for i, group in enumerate(merged):
        text = " ".join(group).strip()
        if len(text) < 10:
            continue
        chunks.append(
            Chunk(
                chunk_id=f"D-{parent_id}-{i}",
                parent_id=parent_id,
                query_id=query_id,
                language=language,
                strategy="semantic",
                chunk_start=i,
                chunk_end=i + len(group),
                text=text,
            )
        )
    return chunks


# ── Strategy E — Parent-child hierarchical ────────────────────────────────────

def strategy_e_parent_child(
    passage: str,
    parent_id: str,
    query_id: str,
    language: str,
    parent_max_tokens: int = 400,
    child_max_tokens: int = 100,
    child_overlap: int = 20,
) -> tuple[list[Chunk], list[Chunk]]:
    """
    Returns (parent_chunks, child_chunks).
    Retrieval uses child chunks; generation receives parent context.
    """
    tokens = _tokenize_simple(passage)
    if not tokens:
        return [], []

    # Parent chunks (~350-400 tokens)
    parent_windows = _chunk_tokens(tokens, parent_max_tokens, parent_max_tokens // 4)

    parent_chunks = []
    child_chunks = []
    parent_char_pos = 0

    for p_idx, p_tokens in enumerate(parent_windows):
        p_text = " ".join(p_tokens)
        if len(p_text) < 10:
            continue

        p_chunk = Chunk(
            chunk_id=f"E-parent-{parent_id}-{p_idx}",
            parent_id=parent_id,
            query_id=query_id,
            language=language,
            strategy="parent_child",
            chunk_start=parent_char_pos,
            chunk_end=parent_char_pos + len(p_text),
            text=p_text,
        )
        parent_chunks.append(p_chunk)

        # Child chunks from this parent (sentence-aware for better boundaries)
        parent_sents = _split_sentences(p_text)
        if parent_sents:
            child_groups = _chunk_tokens(
                _tokenize_simple(p_text), child_max_tokens, child_overlap
            )
            for c_idx, c_tokens in enumerate(child_groups):
                c_text = " ".join(c_tokens)
                if len(c_text) < 10:
                    continue
                child_chunks.append(
                    Chunk(
                        chunk_id=f"E-child-{parent_id}-{p_idx}-{c_idx}",
                        parent_id=parent_id,
                        query_id=query_id,
                        language=language,
                        strategy="parent_child",
                        chunk_start=parent_char_pos,
                        chunk_end=parent_char_pos + len(c_text),
                        text=c_text,
                    )
                )

        parent_char_pos += len(p_text)

    return parent_chunks, child_chunks


# ── Dispatcher ────────────────────────────────────────────────────────────────

def chunk_passage(
    passage: str,
    parent_id: str,
    query_id: str,
    language: str,
    strategy: str,
    embedder=None,
) -> list[Chunk]:
    """Route to the appropriate chunking strategy."""
    strategy = strategy.lower().strip()

    if strategy == "canonical":
        return strategy_a_canonical(passage, parent_id, query_id, language)
    elif strategy == "sentence_window":
        return strategy_b_sentence_windows(passage, parent_id, query_id, language)
    elif strategy == "fixed_token":
        return strategy_c_fixed_token(passage, parent_id, query_id, language)
    elif strategy == "semantic":
        return strategy_d_semantic(passage, parent_id, query_id, language, embedder=embedder)
    elif strategy == "parent_child":
        _, children = strategy_e_parent_child(passage, parent_id, query_id, language)
        return children
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
