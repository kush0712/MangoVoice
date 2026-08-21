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
        self._meta: Optional[list[dict]] = None          # parallel metadata per row
        # In-memory BM25 index (populated by _build_bm25_index)
        # Uses scipy sparse matrix for vectorised O(N) scoring with zero disk IO
        self._bm25_tf: Optional[object] = None    # scipy csr_matrix (N, V) float32
        self._bm25_idf: Optional[np.ndarray] = None  # (V,) IDF weights
        self._bm25_vocab: Optional[dict] = None   # token → column index
        self._bm25_dl_norm: Optional[np.ndarray] = None  # (N, 1) doc-length normalisation

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

        One-time startup cost (~2s to read ~98 MB from disk).  After this every
        dense_search call is a 2-3 ms matrix-vector dot product with no disk IO.
        """
        try:
            n = self._table.count_rows()
            logger.info("Loading %d vectors into RAM (~%.0f MB)...", n, n * 384 * 4 / 1e6)
            t0 = time.perf_counter()

            # Read all rows as an Arrow table (memory-efficient columnar format)
            arrow = self._table.to_arrow()

            # Build (N, 384) float32 and L2-normalise each row so cosine similarity
            # = dot product (avoids per-query norm computation)
            vecs = np.array(arrow["vector"].to_pylist(), dtype=np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            self._vec_normed = (vecs / norms).astype(np.float32)

            # Parallel metadata list — one dict per row, same order as _vec_normed
            _META_KEYS = ["chunk_id", "parent_id", "query_id", "language",
                          "strategy", "chunk_start", "chunk_end", "text"]
            cols = {k: arrow[k].to_pylist() for k in _META_KEYS}
            self._meta = [{k: cols[k][i] for k in _META_KEYS} for i in range(n)]

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
            self._meta = None

    def _build_bm25_index(self) -> None:
        """
        Build a scipy sparse BM25 Okapi index over all chunk texts.

        One-time startup cost (~2-3s). After this, every bm25_search call is
        5-7ms with zero disk IO: tokenize query → slice sparse TF columns →
        vectorised numpy BM25 scoring.

        Memory: ~10MB sparse data + ~1MB IDF array — negligible vs the 98MB
        vector matrix already loaded.

        BM25 Okapi parameters (k1=1.5, b=0.75) are the standard defaults
        — same values Tantivy uses by default in LanceDB FTS.
        """
        if self._meta is None:
            return

        try:
            from scipy import sparse as sp

            t0 = time.perf_counter()
            k1, b = 1.5, 0.75
            texts = [m["text"] or "" for m in self._meta]
            N = len(texts)

            # Build vocabulary and per-doc term frequencies
            vocab: dict[str, int] = {}
            all_tfs: list[Counter] = []
            doc_lens = np.empty(N, dtype=np.float32)

            for i, text in enumerate(texts):
                tokens = _bm25_tokenize(text)
                doc_lens[i] = len(tokens)
                tf = Counter(tokens)
                for tok in tf:
                    if tok not in vocab:
                        vocab[tok] = len(vocab)
                all_tfs.append(tf)

            V = len(vocab)
            avg_dl = float(doc_lens.mean()) if N > 0 else 1.0

            # Build sparse TF matrix (N, V) in float32
            rows, cols_idx, data = [], [], []
            for doc_idx, tf in enumerate(all_tfs):
                for tok, cnt in tf.items():
                    if tok in vocab:
                        rows.append(doc_idx)
                        cols_idx.append(vocab[tok])
                        data.append(float(cnt))

            tf_mat = sp.csr_matrix(
                (data, (rows, cols_idx)), shape=(N, V), dtype=np.float32
            )

            # Precompute IDF: log((N - df + 0.5) / (df + 0.5) + 1)
            df = np.array((tf_mat > 0).sum(axis=0), dtype=np.float32).flatten()
            idf = np.log((N - df + 0.5) / (df + 0.5) + 1).astype(np.float32)

            # Precompute per-doc BM25 denominator factor: k1*(1 - b + b*dl/avg_dl)
            # Shape (N, 1) for broadcasting against (N, Q) TF slices
            dl_factor = (k1 * (1 - b + b * doc_lens / avg_dl)).reshape(-1, 1).astype(np.float32)

            self._bm25_vocab = vocab
            self._bm25_tf = tf_mat
            self._bm25_idf = idf
            self._bm25_dl_norm = dl_factor
            self._bm25_k1 = k1

            elapsed = (time.perf_counter() - t0) * 1000
            logger.info(
                "In-memory BM25 index ready: vocab=%d terms, nnz=%d, built in %.0f ms",
                V, tf_mat.nnz, elapsed,
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
        if norm_q == 0:
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
            meta = self._meta[idx]
            sim = float(sims[idx])
            strategy_str = meta.get("strategy") or "parent_child"
            try:
                strategy = ChunkingStrategy(strategy_str)
            except ValueError:
                strategy = ChunkingStrategy.PARENT_CHILD

            sources.append(RetrievalSource(
                chunk_id=str(meta["chunk_id"]),
                parent_id=str(meta["parent_id"] or ""),
                score=sim,
                raw_dense_score=sim,   # exact — used by confidence gate + extractive rules
                dense_rank=rank,
                text=str(meta["text"] or ""),
                language=str(meta["language"] or "en"),
                strategy=strategy,
                query_id=str(meta.get("query_id") or "") or None,
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

        Uses in-memory scipy sparse index if available (~5-7ms, zero disk IO).
        Falls back to LanceDB Tantivy FTS only if the in-memory index failed
        to build at startup.
        """
        if not self._ready:
            return _demo_sources(top_k)

        if self._bm25_vocab is not None:
            return self._bm25_search_ram(query_text, top_k)

        return self._bm25_search_lancedb(query_text, top_k)

    def _bm25_search_ram(self, query_text: str, top_k: int) -> list[RetrievalSource]:
        """
        In-memory BM25 Okapi scoring via scipy sparse matrix operations.

        Algorithm (vectorised over all N docs simultaneously):
          1. Tokenize query             ~0.01ms
          2. Look up query term cols    ~0.01ms
          3. Slice TF sub-matrix        ~2ms  (sparse column access)
          4. BM25 score = Σ_t idf(t) * tf(d,t)*(k1+1) / (tf(d,t) + dl_factor)
                                        ~2ms  (numpy broadcast arithmetic)
          5. argpartition top-k + sort  ~1ms
        Total: 5-7ms, zero disk IO.
        """
        t0 = time.perf_counter()

        q_tokens = _bm25_tokenize(query_text)
        q_term_indices = [self._bm25_vocab[t] for t in q_tokens if t in self._bm25_vocab]

        if not q_term_indices:
            # No vocabulary match — return empty (dense search still runs in parallel)
            logger.debug(
                "BM25 RAM: no vocab match for query, returning empty",
                extra={"stage": "bm25_search", "latency_ms": 0.0},
            )
            return []

        # Deduplicate term indices (handles repeated query tokens)
        q_term_indices = list(dict.fromkeys(q_term_indices))

        # Slice TF sub-matrix for query terms: (N, Q) dense float32
        q_tf = self._bm25_tf[:, q_term_indices].toarray()  # type: ignore[union-attr]
        q_idf = self._bm25_idf[q_term_indices]              # (Q,)

        # BM25 Okapi: idf * tf*(k1+1) / (tf + dl_factor)
        k1 = self._bm25_k1
        num = q_tf * (k1 + 1)                      # (N, Q)
        denom = q_tf + self._bm25_dl_norm          # (N, Q) via broadcast
        scores = (num / denom * q_idf).sum(axis=1) # (N,) — sum over query terms

        top_k = min(top_k, len(scores))
        top_idx = np.argpartition(scores, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        sources = []
        for rank, idx in enumerate(top_idx):
            if scores[idx] <= 0:
                break  # skip zero-score tails
            m = self._meta[idx]  # type: ignore[index]
            strategy_str = m.get("strategy") or "parent_child"
            try:
                strategy = ChunkingStrategy(strategy_str)
            except ValueError:
                strategy = ChunkingStrategy.PARENT_CHILD
            sources.append(RetrievalSource(
                chunk_id=str(m["chunk_id"]),
                parent_id=str(m["parent_id"] or ""),
                score=float(scores[idx]),
                raw_dense_score=0.0,  # BM25 score — not a cosine sim
                dense_rank=None,
                bm25_rank=rank,
                text=str(m["text"] or ""),
                language=str(m["language"] or "en"),
                strategy=strategy,
                query_id=str(m.get("query_id") or "") or None,
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

