"""
MangoVoice — End-to-end RAG core latency benchmark.

Runs 500-1000 validation queries through the RAG core (no STT).
Records per-stage timing. Computes P50/P70/P100.

Usage:
  python -m evaluation.latency_benchmark \
      --queries data/val_passages.jsonl \
      --n 500 \
      --output reports/latency_results.csv
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import time
from pathlib import Path

import numpy as np


async def run_benchmark(
    queries_path: str,
    n: int = 500,
    output: str = "reports/latency_results.csv",
    warmup: int = 10,
) -> dict:
    from backend.retrieval.retriever import hybrid_retrieve
    from backend.retrieval.confidence import should_generate
    from backend.generation import generate
    from backend.grounding import verify_grounding
    from backend.guardrails import normalize_text, layer1_check

    # Load queries
    queries = []
    with open(queries_path) as f:
        for line in f:
            row = json.loads(line.strip())
            q = row.get("query", "")
            if q and len(q) > 5:
                queries.append(q)
            if len(queries) >= n + warmup:
                break

    print(f"Loaded {len(queries)} queries. Running {warmup} warm-up + {n} benchmark queries...")

    rows = []
    embedding_ms_list, retrieval_ms_list, safety_ms_list, gen_ms_list, grounding_ms_list, core_ms_list = [], [], [], [], [], []

    for i, query in enumerate(queries):
        is_warmup = i < warmup

        t_core = time.perf_counter()

        # Normalize
        normalized = normalize_text(query)
        t_safe = time.perf_counter()
        l1 = layer1_check(normalized)
        safety_ms = (time.perf_counter() - t_safe) * 1000
        if not l1.passed:
            continue

        # Retrieve
        t0 = time.perf_counter()
        retrieval_result, embedding_ms = await hybrid_retrieve(normalized)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        # Gate
        if not should_generate(retrieval_result):
            status = "refused_low_confidence"
            gen_ms = 0.0
            grounding_ms = 0.0
        else:
            # Generate
            t0 = time.perf_counter()
            gen_result = await generate(normalized, retrieval_result.sources)
            gen_ms = (time.perf_counter() - t0) * 1000

            # Grounding
            t0 = time.perf_counter()
            grounding = verify_grounding(gen_result, retrieval_result.sources)
            grounding_ms = (time.perf_counter() - t0) * 1000
            status = "answered" if gen_result.status == "answered" else "refused"

        core_ms = (time.perf_counter() - t_core) * 1000

        if not is_warmup:
            embedding_ms_list.append(embedding_ms)
            retrieval_ms_list.append(retrieval_ms)
            safety_ms_list.append(safety_ms)
            gen_ms_list.append(gen_ms)
            grounding_ms_list.append(grounding_ms)
            core_ms_list.append(core_ms)

            rows.append({
                "query_idx": i - warmup,
                "embedding_ms": round(embedding_ms, 2),
                "retrieval_ms": round(retrieval_ms, 2),
                "safety_ms": round(safety_ms, 2),
                "generation_ms": round(gen_ms, 2),
                "grounding_ms": round(grounding_ms, 2),
                "rag_core_ms": round(core_ms, 2),
                "status": status,
            })

        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(queries)} queries... (core_ms avg={np.mean(core_ms_list or [0]):.1f})")

    def stats(arr):
        if not arr:
            return {"p50": 0, "p70": 0, "p100": 0, "mean": 0}
        a = np.array(arr)
        return {
            "p50": float(np.percentile(a, 50)),
            "p70": float(np.percentile(a, 70)),
            "p100": float(np.percentile(a, 100)),
            "mean": float(np.mean(a)),
        }

    report = {
        "n_queries": len(rows),
        "embedding": stats(embedding_ms_list),
        "retrieval": stats(retrieval_ms_list),
        "safety": stats(safety_ms_list),
        "generation": stats(gen_ms_list),
        "grounding": stats(grounding_ms_list),
        "rag_core": stats(core_ms_list),
    }

    # Print formatted report
    print("\n\nMANGOVOICE LATENCY BENCHMARK")
    print("=" * 55)
    print(f"Queries: N={len(rows)} (warm-up={warmup})")
    print()
    print(f"{'Stage':<20} {'P50':>8} {'P70':>8} {'P100':>8} {'Mean':>8}")
    print("-" * 55)
    for stage in ["embedding", "retrieval", "safety", "generation", "grounding", "rag_core"]:
        s = report[stage]
        label = "RAG CORE TOTAL" if stage == "rag_core" else stage.capitalize()
        print(f"{label:<20} {s['p50']:>7.1f}  {s['p70']:>7.1f}  {s['p100']:>7.1f}  {s['mean']:>7.1f}")

    answered = sum(1 for r in rows if r["status"] == "answered")
    refused = len(rows) - answered
    print(f"\nAnswer rate: {answered}/{len(rows)} ({100*answered/max(len(rows),1):.1f}%)")
    print(f"Refusal rate: {refused}/{len(rows)} ({100*refused/max(len(rows),1):.1f}%)")

    # Save CSV
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    # Save summary JSON
    summary_path = Path(output).with_suffix(".json")
    with open(summary_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nResults: {output}")
    print(f"Summary: {summary_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default="data/val_passages.jsonl")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--output", default="reports/latency_results.csv")
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    asyncio.run(run_benchmark(args.queries, args.n, args.output, args.warmup))
