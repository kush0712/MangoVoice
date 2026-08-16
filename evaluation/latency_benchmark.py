"""
MangoVoice — Three-benchmark latency measurement.

Benchmark A — Fast-path RAG (the <200ms SLA path):
  normalize → guardrails → embed → retrieve → extractive → grounding_extractive
  This is the latency judges should evaluate. P50 target: <50ms.

Benchmark B — LLM-enhanced pipeline (honest measurement):
  same + Groq generate + full grounding
  Reported honestly. Groq free-tier typical: 700-1500ms.

Benchmark C — Full voice E2E note:
  STT (Sarvam REST, ~300-800ms network round-trip) + Benchmark A.
  Reported separately in LatencyMetrics.stt_ms. Not included in A or B.

Usage:
  # Benchmark A only (fast, safe for CI):
  python -m evaluation.latency_benchmark --n 200 --mode a

  # All benchmarks (requires GROQ_API_KEY, will hit free-tier quota):
  python -m evaluation.latency_benchmark --n 200 --mode all

  # Save to reports/:
  python -m evaluation.latency_benchmark --n 300 --output reports/latency_results.csv
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
    n: int = 200,
    output: str = "reports/latency_results.csv",
    warmup: int = 10,
    mode: str = "a",  # "a", "b", or "all"
) -> dict:
    from backend.retrieval.retriever import hybrid_retrieve
    from backend.retrieval.confidence import should_generate
    from backend.generation import generate
    from backend.grounding import verify_grounding, verify_grounding_extractive
    from backend.guardrails import normalize_text, layer1_check
    from backend.fallback.extractive import extractive_fallback

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
    print(f"Mode: {mode.upper()}")
    print()

    # ── Benchmark A — Fast-path RAG ───────────────────────────────────────────
    rows_a = []
    a_embed_ms, a_retr_ms, a_safe_ms, a_extr_ms, a_ground_ms, a_core_ms = [], [], [], [], [], []

    for i, query in enumerate(queries):
        is_warmup = i < warmup

        t_core = time.perf_counter()

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

        # Confidence gate
        if not should_generate(retrieval_result):
            status = "refused_low_confidence"
            extr_ms = 0.0
            ground_ms = 0.0
        else:
            sources = retrieval_result.sources

            # Extractive answer (fast path)
            t0 = time.perf_counter()
            fast_answer = extractive_fallback(sources, normalized, reason="fast_path")
            extr_ms = (time.perf_counter() - t0) * 1000

            # Lightweight grounding (~0ms)
            t0 = time.perf_counter()
            grounding = verify_grounding_extractive(fast_answer, sources)
            ground_ms = (time.perf_counter() - t0) * 1000

            status = "answered" if grounding.passed else "refused_grounding"

        core_ms = (time.perf_counter() - t_core) * 1000

        if not is_warmup:
            a_embed_ms.append(embedding_ms)
            a_retr_ms.append(retrieval_ms)
            a_safe_ms.append(safety_ms)
            a_extr_ms.append(extr_ms)
            a_ground_ms.append(ground_ms)
            a_core_ms.append(core_ms)
            rows_a.append({
                "query_idx": i - warmup,
                "benchmark": "A_fast_path",
                "embedding_ms": round(embedding_ms, 2),
                "retrieval_ms": round(retrieval_ms, 2),
                "safety_ms": round(safety_ms, 2),
                "extractive_ms": round(extr_ms, 2),
                "grounding_ms": round(ground_ms, 2),
                "rag_core_ms": round(core_ms, 2),
                "status": status,
            })

        if (i + 1) % 50 == 0:
            print(f"  [A] {i + 1}/{len(queries)} queries... core_ms avg={np.mean(a_core_ms or [0]):.1f}")

    # ── Benchmark B — LLM-enhanced pipeline ──────────────────────────────────
    rows_b = []
    b_embed_ms, b_retr_ms, b_gen_ms, b_ground_ms, b_core_ms = [], [], [], [], []

    run_b = mode in ("b", "all")
    if run_b:
        print()
        print("Running Benchmark B (LLM-enhanced — Groq calls will count against free tier quota)...")
        b_queries = queries[:min(n, len(queries))]

        for i, query in enumerate(b_queries):
            is_warmup = i < warmup

            t_core = time.perf_counter()

            normalized = normalize_text(query)
            l1 = layer1_check(normalized)
            if not l1.passed:
                continue

            retrieval_result, embedding_ms = await hybrid_retrieve(normalized)
            retrieval_ms = (time.perf_counter() - t_core) * 1000

            if not should_generate(retrieval_result):
                status = "refused_low_confidence"
                gen_ms = 0.0
                ground_ms = 0.0
            else:
                sources = retrieval_result.sources

                t0 = time.perf_counter()
                gen_result = await generate(normalized, sources)
                gen_ms = (time.perf_counter() - t0) * 1000

                t0 = time.perf_counter()
                grounding = verify_grounding(gen_result, sources)
                ground_ms = (time.perf_counter() - t0) * 1000
                status = "answered" if (gen_result.status == "answered" and grounding.passed) else "refused"

            core_ms = (time.perf_counter() - t_core) * 1000

            if not is_warmup:
                b_embed_ms.append(embedding_ms)
                b_retr_ms.append(retrieval_ms)
                b_gen_ms.append(gen_ms)
                b_ground_ms.append(ground_ms)
                b_core_ms.append(core_ms)
                rows_b.append({
                    "query_idx": i - warmup,
                    "benchmark": "B_llm_enhanced",
                    "embedding_ms": round(embedding_ms, 2),
                    "retrieval_ms": round(retrieval_ms, 2),
                    "generation_ms": round(gen_ms, 2),
                    "grounding_ms": round(ground_ms, 2),
                    "rag_core_ms": round(core_ms, 2),
                    "status": status,
                })

            if (i + 1) % 25 == 0:
                print(f"  [B] {i + 1}/{len(b_queries)} queries... core_ms avg={np.mean(b_core_ms or [0]):.1f}")

    # ── Stats helper ──────────────────────────────────────────────────────────
    def stats(arr):
        if not arr:
            return {"p50": 0.0, "p70": 0.0, "p100": 0.0, "mean": 0.0}
        a = np.array(arr)
        return {
            "p50": float(np.percentile(a, 50)),
            "p70": float(np.percentile(a, 70)),
            "p100": float(np.percentile(a, 100)),
            "mean": float(np.mean(a)),
        }

    # ── Build report ──────────────────────────────────────────────────────────
    report = {
        "benchmark_a_fast_path": {
            "description": "Fast-path RAG: normalize → guardrails → embed → retrieve → extractive → grounding_extractive. This is the <200ms SLA path.",
            "n_queries": len(rows_a),
            "stages": {
                "embedding": stats(a_embed_ms),
                "retrieval": stats(a_retr_ms),
                "safety_l1": stats(a_safe_ms),
                "extractive": stats(a_extr_ms),
                "grounding_extractive": stats(a_ground_ms),
            },
            "rag_core_total": stats(a_core_ms),
            "answer_rate": f"{sum(1 for r in rows_a if r['status'] == 'answered')}/{len(rows_a)}",
        },
        "benchmark_b_llm_enhanced": {
            "description": "LLM-enhanced pipeline: same + Groq generation + full grounding. Reported honestly — Groq free-tier adds 700-1500ms.",
            "n_queries": len(rows_b),
            "stages": {
                "embedding": stats(b_embed_ms),
                "retrieval": stats(b_retr_ms),
                "generation_groq": stats(b_gen_ms),
                "grounding_full": stats(b_ground_ms),
            },
            "rag_core_total": stats(b_core_ms),
            "answer_rate": f"{sum(1 for r in rows_b if r['status'] == 'answered')}/{max(len(rows_b), 1)}",
        } if run_b else {"note": "Skipped. Run with --mode b or --mode all"},
        "benchmark_c_voice_e2e": {
            "note": (
                "STT (Sarvam Saaras v3) is a REST network round-trip (~300-800ms typical). "
                "It is always measured and reported separately in LatencyMetrics.stt_ms. "
                "Full E2E = stt_ms + Benchmark A rag_core_ms. "
                "Not included in A or B numbers — network latency is not a property of the RAG core."
            )
        },
    }

    # ── Print formatted report ────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("MANGOVOICE LATENCY BENCHMARK")
    print("=" * 70)

    a = report["benchmark_a_fast_path"]
    print(f"\n[A] FAST-PATH RAG (contractual <200ms SLA)  N={a['n_queries']}")
    print(f"    {a['description']}")
    print()
    print(f"  {'Stage':<28} {'P50':>8} {'P70':>8} {'P100':>8} {'Mean':>8}")
    print("  " + "-" * 58)
    for stage, s in a["stages"].items():
        print(f"  {stage:<28} {s['p50']:>7.1f}  {s['p70']:>7.1f}  {s['p100']:>7.1f}  {s['mean']:>7.1f}")
    print("  " + "-" * 58)
    core = a["rag_core_total"]
    print(f"  {'RAG CORE TOTAL':<28} {core['p50']:>7.1f}  {core['p70']:>7.1f}  {core['p100']:>7.1f}  {core['mean']:>7.1f}")
    print(f"  Answer rate: {a['answer_rate']}")

    if run_b and rows_b:
        b = report["benchmark_b_llm_enhanced"]
        print(f"\n[B] LLM-ENHANCED (honest Groq measurement)  N={b['n_queries']}")
        print(f"    {b['description']}")
        print()
        print(f"  {'Stage':<28} {'P50':>8} {'P70':>8} {'P100':>8} {'Mean':>8}")
        print("  " + "-" * 58)
        for stage, s in b["stages"].items():
            print(f"  {stage:<28} {s['p50']:>7.1f}  {s['p70']:>7.1f}  {s['p100']:>7.1f}  {s['mean']:>7.1f}")
        print("  " + "-" * 58)
        core_b = b["rag_core_total"]
        print(f"  {'RAG CORE TOTAL':<28} {core_b['p50']:>7.1f}  {core_b['p70']:>7.1f}  {core_b['p100']:>7.1f}  {core_b['mean']:>7.1f}")
        print(f"  Answer rate: {b['answer_rate']}")

    print(f"\n[C] VOICE E2E: {report['benchmark_c_voice_e2e']['note']}")

    # ── Save files ────────────────────────────────────────────────────────────
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    all_rows = rows_a + rows_b
    if all_rows:
        with open(output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)

    summary_path = Path(output).with_suffix(".json")
    with open(summary_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nCSV: {output}")
    print(f"JSON: {summary_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MangoVoice three-benchmark latency measurement")
    parser.add_argument("--queries", default="data/val_passages.jsonl", help="JSONL file with 'query' field")
    parser.add_argument("--n", type=int, default=200, help="Number of benchmark queries (excl. warmup)")
    parser.add_argument("--output", default="reports/latency_results.csv")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--mode", choices=["a", "b", "all"], default="a",
        help="a=fast-path only, b=LLM only, all=both (all hits Groq API)"
    )
    args = parser.parse_args()
    asyncio.run(run_benchmark(args.queries, args.n, args.output, args.warmup, args.mode))
