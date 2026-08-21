"""
MangoVoice — LanceDB embedded vector + BM25 store.

Opens a pre-built read-only index at startup.
Provides: dense vector search (in-memory numpy exact search), BM25 full-text search.

Dense search strategy
─────────────────────
At startup _load_into_ram() reads all 63,615 vectors into a pre-normalised
(N, 384) float32 numpy matrix (~98 MB). Every subsequent dense query is a
single matrix-vector dot product: ~2-3 ms with no disk IO and exact cosine
similarity. This avoids two problems with the IVF_PQ index:
  1. Disk IO latency (100+ ms cold) — data is already in RAM.
  2. PQ approximation error in _distance — score is the true cosine sim,
     so the confidence gate and extractive rules always see correct values.
If the RAM load fails (OOM etc.), dense_search falls back to LanceDB ANN.

BM25 search strategy
─────────────────────
At startup _build_bm25_index() builds a scipy sparse TF matrix over all
chunk texts and precomputes IDF weights. Every subsequent BM25 query is:
  1. Tokenize query (regex, ~0.01ms)
  2. Slice relevant columns from sparse TF matrix (.toarray(), ~2-5ms)
  3. Vectorised BM25 Okapi score via numpy (~1-2ms)
  Total: 5-7ms with zero disk IO.

This replaces LanceDB Tantivy FTS which hits disk on Railway's container
filesystem (~300ms wall-clock). The scipy index is read-only and built
from the same texts already in _meta — semantically identical results.
Falls back to LanceDB FTS if scipy build fails (OOM etc.).
"""
from __future__ import annotations

import os
import re
import time
from collections import Counter
from typing import Optional

import numpy as np

from backend.config import settings
from backend.schemas import RetrievalSource, ChunkingStrategy
from backend.telemetry import get_logger

logger = get_logger(__name__)


# ── BM25 tokenizer ────────────────────────────────────────────────────────────
# Unicode-aware: handles English words, Devanagari script, and Hinglish
# (latin + devanagari mixed). Consistent tokenizer used at both index-build
# time and query time ensures correct term matching across all three languages.
_TOKEN_RE = re.compile(r"[\w\u0900-\u097F]+")


def _bm25_tokenize(text: str) -> list[str]:
    """Simple regex tokenizer for BM25. Lower-cases for consistent matching."""
    return _TOKEN_RE.findall(text.lower())

_store_singleton: Optional["LanceDBStore"] = None


class LanceDBStore:
    """Embedded LanceDB store for MangoVoice."""

    def __init__(self) -> None:
        self._db = None
        self._table = None
        self._ready = False
        # In-memory exact search index (populated by _load_into_ram)
        self._vec_normed: Optional[np.ndarray] = None   # (N, 384) L2-normalised float32
        # Columnar metadata lists for fast, low-memory access
        self._meta_chunk_id: Optional[list] = None
        self._meta_parent_id: Optional[list] = None
        self._meta_text: Optional[list] = None
        self._meta_language: Optional[list] = None
        self._meta_strategy: Optional[list] = None
        self._meta_query_id: Optional[list] = None
        # In-memory flat BM25 index (populated by _build_bm25_index)
        self._bm25_vocab: Optional[dict] = None          # token → term_idx
        self._bm25_flat_docs: Optional[np.ndarray] = None # contiguous int32 doc ids
        self._bm25_flat_tfs: Optional[np.ndarray] = None  # contiguous float32 tfs
        self._bm25_offsets: Optional[np.ndarray] = None   # (V,) offsets into flat arrays
        self._bm25_lengths: Optional[np.ndarray] = None   # (V,) term posting counts
        self._bm25_idf: Optional[np.ndarray] = None       # (V,) IDF weights
        self._bm25_dl_norm: Optional[np.ndarray] = None   # (N,) doc-length normalisation
        self._bm25_k1: float = 1.5

    def _load(self) -> None:
        if self._table is not None:
            return

        index_path = settings.index_path
        if not os.path.exists(index_path):
            logger.warning(
                "LanceDB index not found at %s — running in demo/stub mode", index_path
            )
            self._ready = False
            return

        try:
            import lancedb

            self._db = lancedb.connect(index_path)
            self._table = self._db.open_table(settings.index_table)
            self._ready = True
            logger.info(
                "LanceDB index opened: %s / %s (%d rows)",
                index_path,
                settings.index_table,
                self._table.count_rows(),
            )
        except Exception as exc:
            logger.error("Failed to open LanceDB index: %s", exc)
            self._ready = False
            return

        # Load all vectors into RAM for fast exact in-memory search
        self._load_into_ram()
        # Build in-memory BM25 index for zero-disk-IO keyword search
        self._build_bm25_index()

    def _load_into_ram(self) -> None:
        """
        Read all 384-dim vectors into a pre-normalised (N, 384) numpy matrix.
        Uses zero-copy PyArrow flat buffer conversion (instant, minimal memory).
        """
        try:
            n = self._table.count_rows()
            t0 = time.perf_counter()

            # Read Arrow table from LanceDB
            arrow = self._table.to_arrow()

            # Zero-copy conversion of FixedSizeListArray to (N, 384) float32
            flat = np.array(arrow["vector"].combine_chunks().values, copy=False)
            vecs = flat.reshape((n, 384)).astype(np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            self._vec_normed = (vecs / norms).astype(np.float32)

            # Compact columnar metadata lists (minimal memory, zero per-row dict overhead)
            self._meta_chunk_id = arrow["chunk_id"].to_pylist()
            self._meta_parent_id = arrow["parent_id"].to_pylist()
            self._meta_text = arrow["text"].to_pylist()
            self._meta_language = arrow["language"].to_pylist()
            self._meta_strategy = arrow["strategy"].to_pylist()
            self._meta_query_id = arrow["query_id"].to_pylist()

            elapsed = (time.perf_counter() - t0) * 1000
            logger.info(
                "In-memory vector index ready: %d vectors, %.1f MB, loaded in %.0f ms",
                n, self._vec_normed.nbytes / 1e6, elapsed,
            )
        except Exception as exc:
            logger.warning(
                "Failed to load vectors into RAM — dense search will use LanceDB ANN fallback: %s", exc
            )
            self._vec_normed = None
            self._meta_chunk_id = None

    def _build_bm25_index(self) -> None:
        """
        Build a high-performance contiguous flat inverted BM25 Okapi index.
        Memory: ~15 MB total. Query lookup: ~0.15ms.
        """
        if self._meta_text is None:
            return

        try:
            t0 = time.perf_counter()
            k1, b = 1.5, 0.75
            texts = self._meta_text
            N = len(texts)

            vocab: dict[str, int] = {}
            doc_lens = np.empty(N, dtype=np.float32)
            doc_tokens_list = []

            for i, text in enumerate(texts):
                toks = _bm25_tokenize(text or "")
                doc_lens[i] = len(toks)
                tf = Counter(toks)
                doc_tokens_list.append(tf)
                for tok in tf:
                    if tok not in vocab:
                        vocab[tok] = len(vocab)

            V = len(vocab)
            term_counts = np.zeros(V, dtype=np.int32)
            for tf in doc_tokens_list:
                for tok in tf:
                    term_counts[vocab[tok]] += 1

            offsets = np.zeros(V, dtype=np.int32)
            offsets[1:] = np.cumsum(term_counts[:-1])
            lengths = term_counts.copy()

            total_postings = int(term_counts.sum())
            flat_docs = np.empty(total_postings, dtype=np.int32)
            flat_tfs = np.empty(total_postings, dtype=np.float32)

            curr_pos = offsets.copy()
            for doc_id, tf in enumerate(doc_tokens_list):
                for tok, cnt in tf.items():
                    v_idx = vocab[tok]
                    pos = curr_pos[v_idx]
                    flat_docs[pos] = doc_id
                    flat_tfs[pos] = float(cnt)
                    curr_pos[v_idx] += 1

            df = lengths.astype(np.float32)
            idf = np.log((N - df + 0.5) / (df + 0.5) + 1).astype(np.float32)
            avg_dl = float(doc_lens.mean()) if N > 0 else 1.0
            dl_factor = (k1 * (1 - b + b * doc_lens / avg_dl)).astype(np.float32)

            self._bm25_vocab = vocab
            self._bm25_flat_docs = flat_docs
            self._bm25_flat_tfs = flat_tfs
            self._bm25_offsets = offsets
            self._bm25_lengths = lengths
            self._bm25_idf = idf
            self._bm25_dl_norm = dl_factor
            self._bm25_k1 = k1

            elapsed = (time.perf_counter() - t0) * 1000
            logger.info(
                "In-memory flat BM25 index ready: vocab=%d terms, postings=%d (%.1f MB), built in %.0f ms",
                V, total_postings, (flat_docs.nbytes + flat_tfs.nbytes) / 1e6, elapsed,
            )
        except Exception as exc:
            logger.warning(
                "Failed to build in-memory BM25 — keyword search will use LanceDB FTS fallback: %s", exc
            )
            self._bm25_vocab = None

    def is_ready(self) -> bool:
        return self._ready

    def dense_search(self, query_vec: np.ndarray, top_k: int) -> list[RetrievalSource]:
        """Exact cosine search. Uses in-memory numpy if available, LanceDB ANN as fallback."""
        if not self._ready:
            return _demo_sources(top_k)

        if self._vec_normed is not None:
            return self._dense_numpy(query_vec, top_k)

        # Fallback: IVF_PQ ANN (used only if _load_into_ram failed)
        return self._dense_lancedb(query_vec, top_k)

    def _dense_numpy(self, query_vec: np.ndarray, top_k: int) -> list[RetrievalSource]:
        """
        2-3 ms exact cosine search via pre-normalised matrix dot product.
        raw_dense_score is the true cosine similarity — no PQ approximation.
        """
        t0 = time.perf_counter()

        norm_q = np.linalg.norm(query_vec)
        if norm_q == 0 or np.isnan(norm_q):
            return []
        q_normed = (query_vec / norm_q).astype(np.float32)

        # Single BLAS matrix-vector multiply: (N, 384) @ (384,) → (N,) cosine sims
        sims = self._vec_normed @ q_normed

        # argpartition is O(N); then sort only the top-k slice
        top_k = min(top_k, len(sims))
        top_idx = np.argpartition(sims, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]

        sources = []
        for rank, idx in enumerate(top_idx):
            sim = float(sims[idx])
            strategy_str = self._meta_strategy[idx] or "parent_child"
            try:
                strategy = ChunkingStrategy(strategy_str)
            except ValueError:
                strategy = ChunkingStrategy.PARENT_CHILD

            sources.append(RetrievalSource(
                chunk_id=str(self._meta_chunk_id[idx]),
                parent_id=str(self._meta_parent_id[idx] or ""),
                score=sim,
                raw_dense_score=sim,   # exact — used by confidence gate + extractive rules
                dense_rank=rank,
                text=str(self._meta_text[idx] or ""),
                language=str(self._meta_language[idx] or "en"),
                strategy=strategy,
                query_id=str(self._meta_query_id[idx] or "") or None,
            ))

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            "Dense search (numpy): %d results in %.2f ms", len(sources), latency_ms,
            extra={"stage": "dense_search", "latency_ms": round(latency_ms, 1)},
        )
        return sources

    def _dense_lancedb(self, query_vec: np.ndarray, top_k: int) -> list[RetrievalSource]:
        """
        Fallback IVF_PQ ANN search via LanceDB (used only if RAM load failed).
        Recomputes exact cosine from stored vectors to correct PQ approximation error.
        """
        t0 = time.perf_counter()
        try:
            results = (
                self._table.search(query_vec.tolist())
                .metric("cosine")
                .nprobes(8)
                .limit(top_k)
                .to_list()
            )
        except Exception as exc:
            logger.error("LanceDB dense search failed: %s", exc)
            return []

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            "Dense search (LanceDB fallback): %d results", len(results),
            extra={"stage": "dense_search", "latency_ms": round(latency_ms, 1)},
        )

        # Correct PQ approximation: recompute exact cosine from stored float32 vectors
        sources = []
        norm_q = np.linalg.norm(query_vec)  # constant — compute once
        for i, r in enumerate(results):
            source = _row_to_source(r, rank=i)
            if "vector" in r and norm_q > 0:
                v_stored = np.array(r["vector"], dtype=np.float32)
                norm_v = np.linalg.norm(v_stored)
                if norm_v > 0:
                    true_sim = float(np.dot(query_vec, v_stored) / (norm_q * norm_v))
                    source.raw_dense_score = true_sim
                    source.score = true_sim
            sources.append(source)

        sources.sort(key=lambda s: s.raw_dense_score, reverse=True)
        for i, s in enumerate(sources):
            s.dense_rank = i
        return sources

    def bm25_search(self, query_text: str, top_k: int) -> list[RetrievalSource]:
        """
        BM25 full-text search.
        Uses in-memory NumPy flat index (~0.15ms, zero disk IO).
        Falls back to LanceDB Tantivy FTS only if the in-memory index failed at startup.
        """
        if not self._ready:
            return _demo_sources(top_k)

        if self._bm25_vocab is not None:
            return self._bm25_search_ram(query_text, top_k)

        return self._bm25_search_lancedb(query_text, top_k)

    def _bm25_search_ram(self, query_text: str, top_k: int) -> list[RetrievalSource]:
        """
        In-memory BM25 Okapi scoring via flat NumPy inverted index.
        Query lookup: ~0.15ms, zero disk IO.
        """
        t0 = time.perf_counter()

        q_tokens = _bm25_tokenize(query_text)
        q_term_indices = [self._bm25_vocab[t] for t in q_tokens if t in self._bm25_vocab]

        if not q_term_indices:
            logger.debug(
                "BM25 RAM: no vocab match for query, returning empty",
                extra={"stage": "bm25_search", "latency_ms": 0.0},
            )
            return []

        # Deduplicate term indices
        q_term_indices = list(dict.fromkeys(q_term_indices))

        # Accumulate scores directly into float32 array
        scores = np.zeros(len(self._meta_chunk_id), dtype=np.float32)
        k1 = self._bm25_k1

        for term_idx in q_term_indices:
            off = self._bm25_offsets[term_idx]
            length = self._bm25_lengths[term_idx]
            docs = self._bm25_flat_docs[off : off + length]
            tfs = self._bm25_flat_tfs[off : off + length]
            term_idf = self._bm25_idf[term_idx]
            
            num = tfs * (k1 + 1)
            denom = tfs + self._bm25_dl_norm[docs]
            term_scores = (num / denom) * term_idf
            
            scores[docs] += term_scores

        top_k = min(top_k, len(scores))
        top_idx = np.argpartition(scores, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        sources = []
        for rank, idx in enumerate(top_idx):
            if scores[idx] <= 0:
                break
            strategy_str = self._meta_strategy[idx] or "parent_child"
            try:
                strategy = ChunkingStrategy(strategy_str)
            except ValueError:
                strategy = ChunkingStrategy.PARENT_CHILD
            sources.append(RetrievalSource(
                chunk_id=str(self._meta_chunk_id[idx]),
                parent_id=str(self._meta_parent_id[idx] or ""),
                score=float(scores[idx]),
                raw_dense_score=0.0,
                dense_rank=None,
                bm25_rank=rank,
                text=str(self._meta_text[idx] or ""),
                language=str(self._meta_language[idx] or "en"),
                strategy=strategy,
                query_id=str(self._meta_query_id[idx] or "") or None,
            ))

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            "BM25 search (RAM): %d results in %.2f ms", len(sources), latency_ms,
            extra={"stage": "bm25_search", "latency_ms": round(latency_ms, 1)},
        )
        return sources

    def _bm25_search_lancedb(self, query_text: str, top_k: int) -> list[RetrievalSource]:
        """Fallback BM25 via LanceDB Tantivy FTS (used only if in-memory index failed)."""
        t0 = time.perf_counter()
        try:
            results = (
                self._table.search(query_text, query_type="fts")
                .limit(top_k)
                .to_list()
            )
        except Exception as exc:
            logger.error("BM25 search (LanceDB fallback) failed: %s", exc)
            return []

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            "BM25 search (LanceDB): %d results", len(results),
            extra={"stage": "bm25_search", "latency_ms": round(latency_ms, 1)},
        )
        return [_row_to_source(r, rank=i) for i, r in enumerate(results)]


def _row_to_source(row: dict, rank: int = 0) -> RetrievalSource:
    score = float(row.get("_distance", 0.0))
    # Cosine distance → similarity: similarity = 1 - distance
    if score <= 1.0:
        score = max(0.0, 1.0 - score)

    strategy_str = row.get("strategy", "parent_child")
    try:
        strategy = ChunkingStrategy(strategy_str)
    except ValueError:
        strategy = ChunkingStrategy.PARENT_CHILD

    return RetrievalSource(
        chunk_id=str(row.get("chunk_id", f"chunk-{rank}")),
        parent_id=str(row.get("parent_id", "")),
        score=score,
        raw_dense_score=score,  # preserve cosine similarity before RRF overwrites .score
        dense_rank=rank,
        text=str(row.get("text", "")),
        language=str(row.get("language", "en")),
        strategy=strategy,
        query_id=str(row.get("query_id", "")) or None,
    )


def _demo_sources(n: int) -> list[RetrievalSource]:
    """Stub sources returned when no index is available (demo mode)."""
    demo_texts = [
        "The Manhattan Project was a research and development undertaking during World War II that produced the first nuclear weapons.",
        "India gained independence from British rule on August 15, 1947, following a long independence movement led by figures like Mahatma Gandhi.",
        "The Goa Liberation was completed on December 19, 1961, when Indian Armed Forces ended Portuguese colonial rule.",
        "Artificial intelligence refers to the simulation of human intelligence in machines programmed to think and learn.",
        "Retrieval-augmented generation (RAG) combines information retrieval with language model generation for grounded answers.",
    ]
    return [
        RetrievalSource(
            chunk_id=f"demo-chunk-{i}",
            parent_id=f"demo-parent-{i}",
            score=0.85 - i * 0.05,
            dense_rank=i,
            text=demo_texts[i % len(demo_texts)],
            language="en",
            strategy=ChunkingStrategy.PARENT_CHILD,
        )
        for i in range(min(n, len(demo_texts)))
    ]


def get_store() -> LanceDBStore:
    global _store_singleton
    if _store_singleton is None:
        _store_singleton = LanceDBStore()
        _store_singleton._load()
    return _store_singleton

