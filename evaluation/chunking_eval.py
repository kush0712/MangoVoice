"""
MangoVoice — Chunking strategy evaluation.

Evaluates all 5 strategies against the MSMARCO-XI validation split.
Uses is_selected labels for ground truth.

Metrics: Recall@5, Recall@10, MRR@10, P50/P70/P100 retrieval latency

Usage:
  python -m evaluation.chunking_eval \
      --val-data data/val_passages.jsonl \
      --output reports/chunking_results.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np


STRATEGIES = ["canonical", "sentence_window", "fixed_token", "semantic", "parent_child"]


def recall_at_k(retrieved_parent_ids: list[str], gold_parent_ids: set[str], k: int) -> float:
    top_k = retrieved_parent_ids[:k]
    return 1.0 if any(pid in gold_parent_ids for pid in top_k) else 0.0


def mrr_at_k(retrieved_parent_ids: list[str], gold_parent_ids: set[str], k: int = 10) -> float:
    for rank, pid in enumerate(retrieved_parent_ids[:k], start=1):
        if pid in gold_parent_ids:
            return 1.0 / rank
    return 0.0


def evaluate_strategy(
    strategy: str,
    passages: list[dict],
    val_queries: list[dict],
    embedding_model: str,
    top_k: int = 10,
) -> dict:
    """Build an in-memory index for strategy and evaluate on val_queries."""
    from fastembed import TextEmbedding
    from ingestion.chunkers import chunk_passage

    print(f"\n  Building {strategy} index...")
    embedder = TextEmbedding(model_name=embedding_model, cache_dir="data/fastembed_cache")

    class EmbedWrap:
        def embed(self, texts): return np.array(list(embedder.embed(texts)), dtype=np.float32)

    ew = EmbedWrap()

    # Chunk all passages
    all_chunks = []
    for p in passages:
        chunks = chunk_passage(
            passage=p["text"],
            parent_id=p["passage_id"],
            query_id=p["query_id"],
            language=p["language"],
            strategy=strategy,
            embedder=ew if strategy == "semantic" else None,
        )
        all_chunks.extend(chunks)

    texts = [c.text for c in all_chunks]
    print(f"  {len(all_chunks)} chunks — embedding...")

    # Embed chunks
    vectors = np.array(list(embedder.embed(texts)), dtype=np.float32)

    # Evaluate queries
    recalls_5, recalls_10, mrrs, latencies = [], [], [], []

    for qrow in val_queries:
        query = qrow.get("query", "")
        # Gold: passage IDs where is_selected=1
        gold_ids = set(qrow.get("gold_parent_ids", []))
        if not gold_ids or not query:
            continue

        t0 = time.perf_counter()
        q_vec = np.array(list(embedder.embed([query]))[0], dtype=np.float32)

        # Cosine similarity
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norm_vecs = vectors / (norms + 1e-8)
        q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-8)
        sims = norm_vecs @ q_norm
        top_indices = np.argsort(-sims)[:top_k]
        latency_ms = (time.perf_counter() - t0) * 1000

        retrieved_parent_ids = [all_chunks[i].parent_id for i in top_indices]

        recalls_5.append(recall_at_k(retrieved_parent_ids, gold_ids, 5))
        recalls_10.append(recall_at_k(retrieved_parent_ids, gold_ids, 10))
        mrrs.append(mrr_at_k(retrieved_parent_ids, gold_ids, 10))
        latencies.append(latency_ms)

    latencies_arr = np.array(latencies) if latencies else np.array([0.0])

    return {
        "strategy": strategy,
        "n_chunks": len(all_chunks),
        "n_queries": len(recalls_5),
        "recall_at_5": float(np.mean(recalls_5)) if recalls_5 else 0.0,
        "recall_at_10": float(np.mean(recalls_10)) if recalls_10 else 0.0,
        "mrr_at_10": float(np.mean(mrrs)) if mrrs else 0.0,
        "latency_p50_ms": float(np.percentile(latencies_arr, 50)),
        "latency_p70_ms": float(np.percentile(latencies_arr, 70)),
        "latency_p100_ms": float(np.percentile(latencies_arr, 100)),
    }


def run_evaluation(
    val_data: str,
    passage_data: str = "data/raw_passages.jsonl",
    output: str = "reports/chunking_results.csv",
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    max_passages: int = 5000,
    max_queries: int = 500,
) -> list[dict]:
    print("Loading passages...")
    passages = []
    with open(passage_data) as f:
        for i, line in enumerate(f):
            if i >= max_passages:
                break
            passages.append(json.loads(line.strip()))

    print(f"Loaded {len(passages)} passages")

    print("Loading validation queries...")
    val_queries = []
    with open(val_data) as f:
        for i, line in enumerate(f):
            if i >= max_queries:
                break
            row = json.loads(line.strip())
            gold_ids = row.get("gold_parent_ids") or []
            if not gold_ids:
                is_sel = row.get("is_selected_list", [])
                qid = row.get("query_id", str(i))
                gold_ids = [f"hi-{qid}-{j}" for j, s in enumerate(is_sel) if s == 1] + [f"en-{qid}-{j}" for j, s in enumerate(is_sel) if s == 1]
            if not gold_ids:
                gold_ids = [f"hi-{row.get('query_id', i)}-0", f"en-{row.get('query_id', i)}-0"]
            val_queries.append({
                "query": row.get("query", ""),
                "gold_parent_ids": gold_ids,
            })

    results = []
    for strategy in STRATEGIES:
        try:
            r = evaluate_strategy(strategy, passages, val_queries, embedding_model)
            results.append(r)
            print(
                f"  {strategy:20s} "
                f"R@5={r['recall_at_5']:.3f} "
                f"R@10={r['recall_at_10']:.3f} "
                f"MRR={r['mrr_at_10']:.3f} "
                f"P50={r['latency_p50_ms']:.1f}ms"
            )
        except Exception as exc:
            print(f"  ERROR evaluating {strategy}: {exc}")

    # Print table
    print("\n\nCHUNKING STRATEGY EVALUATION")
    print("=" * 70)
    print(f"{'Strategy':<22} {'R@5':>6} {'R@10':>6} {'MRR@10':>8} {'P50(ms)':>9}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['strategy']:<22} "
            f"{r['recall_at_5']:>6.3f} "
            f"{r['recall_at_10']:>6.3f} "
            f"{r['mrr_at_10']:>8.3f} "
            f"{r['latency_p50_ms']:>9.1f}"
        )

    # Save CSV
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys() if results else [])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {output}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-data", default="data/val_passages.jsonl")
    parser.add_argument("--passage-data", default="data/raw_passages.jsonl")
    parser.add_argument("--output", default="reports/chunking_results.csv")
    parser.add_argument("--max-passages", type=int, default=5000)
    parser.add_argument("--max-queries", type=int, default=500)
    args = parser.parse_args()
    run_evaluation(
        val_data=args.val_data,
        passage_data=args.passage_data,
        output=args.output,
        max_passages=args.max_passages,
        max_queries=args.max_queries,
    )
