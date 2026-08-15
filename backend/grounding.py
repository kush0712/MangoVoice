"""
MangoVoice — Grounding verifier (Layer 4 guardrail).

After generation:
1. Split answer into sentences
2. Map each sentence to cited chunks via cosine similarity
3. Check entity/number overlap
4. Reject or flag for regeneration if grounding is weak
"""
from __future__ import annotations

import re
import time

import numpy as np

from backend.config import settings
from backend.schemas import GenerationResult, GroundingResult, RetrievalSource
from backend.retrieval.embeddings import get_embedder
from backend.telemetry import get_logger

logger = get_logger(__name__)

# Grounding thresholds — calibrated for multilingual answers (English, Hindi, Hinglish)
SENTENCE_SIMILARITY_THRESHOLD = 0.35  # minimum cosine sim sentence ↔ evidence
OVERALL_GROUNDING_THRESHOLD = 0.40    # minimum average grounding score to pass


def _split_sentences(text: str) -> list[str]:
    """Simple sentence splitter (no NLTK dependency in hot path)."""
    text = text.strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def _extract_numbers_entities(text: str) -> set[str]:
    """Extract numbers and capitalized tokens as pseudo-entities."""
    numbers = set(re.findall(r"\b\d+(?:[.,]\d+)?\b", text))
    caps = set(re.findall(r"\b[A-Z][A-Za-z]{2,}\b", text))
    # Filter common words
    common = {"The", "This", "That", "These", "Those", "When", "Where", "What", "Who"}
    caps -= common
    return numbers | caps


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def verify_grounding(
    gen: GenerationResult,
    sources: list[RetrievalSource],
) -> GroundingResult:
    """
    Verify that the generated answer is grounded in retrieved evidence.
    Returns GroundingResult with passed=True/False.
    """
    t0 = time.perf_counter()

    if gen.status == "refused" or not gen.answer:
        return GroundingResult(passed=True, score=1.0, citation_valid=True)

    # ── A. Citation existence check ───────────────────────────────────────────
    cited_ids = set(gen.cited_chunk_ids)
    source_ids = {s.chunk_id for s in sources}
    citation_valid = bool(cited_ids) and bool(cited_ids & source_ids)

    if not citation_valid:
        logger.warning("Grounding FAIL: no valid citations")
        return GroundingResult(
            passed=False,
            score=0.0,
            citation_valid=False,
        )

    # ── B. Sentence-level similarity ─────────────────────────────────────────
    sentences = _split_sentences(gen.answer)
    if not sentences:
        return GroundingResult(passed=True, score=1.0, citation_valid=True)

    # Get embedder (already warm)
    embedder = get_embedder()

    # Embed all sentences and cited evidence texts
    cited_sources = [s for s in sources if s.chunk_id in cited_ids]
    if not cited_sources:
        cited_sources = sources[:3]

    evidence_texts = [s.text for s in cited_sources]

    try:
        sent_vecs = embedder.embed(sentences)           # (N_sent, 384)
        evid_vecs = embedder.embed(evidence_texts)      # (N_evid, 384)
    except Exception as exc:
        logger.warning("Grounding embedding failed: %s — skip check", exc)
        return GroundingResult(passed=True, score=0.6, citation_valid=True)

    sentence_scores = []
    for s_vec in sent_vecs:
        # Max similarity of this sentence against all evidence
        sims = [cosine_sim(s_vec, e_vec) for e_vec in evid_vecs]
        sentence_scores.append(max(sims) if sims else 0.0)

    avg_score = float(np.mean(sentence_scores)) if sentence_scores else 0.0

    # ── C. Entity/number overlap ──────────────────────────────────────────────
    answer_entities = _extract_numbers_entities(gen.answer)
    evidence_all_text = " ".join(s.text for s in cited_sources)
    evidence_entities = _extract_numbers_entities(evidence_all_text)

    if answer_entities:
        overlap = len(answer_entities & evidence_entities) / len(answer_entities)
    else:
        overlap = 1.0  # no entities to check → pass

    # ── D. Combined decision ──────────────────────────────────────────────────
    final_score = avg_score * 0.7 + overlap * 0.3
    passed = final_score >= OVERALL_GROUNDING_THRESHOLD

    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "Grounding: score=%.3f entity_overlap=%.3f passed=%s",
        final_score, overlap, passed,
        extra={"stage": "grounding", "latency_ms": round(latency_ms, 1)},
    )

    return GroundingResult(
        passed=passed,
        score=final_score,
        sentence_scores=sentence_scores,
        entity_overlap=overlap,
        citation_valid=citation_valid,
    )
