import asyncio
import httpx
import json
import numpy as np
from pathlib import Path
import time

async def run_live_benchmark(n=199, endpoint="https://mango-voice.vercel.app/api/query/text"):
    print(f"Loading {n} queries from data/val_passages.jsonl...")
    queries = []
    with open("data/val_passages.jsonl") as f:
        for line in f:
            row = json.loads(line.strip())
            q = row.get("query", "")
            if q and len(q) > 5:
                queries.append(q)
            if len(queries) >= n:
                break
                
    print(f"Loaded {len(queries)} queries.")
    print(f"Sending requests to {endpoint}...\n")
    
    embed_ms, retr_ms, extr_ms, ground_ms, core_ms = [], [], [], [], []
    answered = 0
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # We process sequentially or with small concurrency to not trigger rate limits/DDoS protection
        for i, query in enumerate(queries):
            try:
                resp = await client.post(
                    endpoint, 
                    json={"text": query, "language": "auto"}
                )
                if resp.status_code == 200:
                    data = resp.json()
                    
                    if data.get("status") == "answered":
                        answered += 1
                        
                    # Extract the server-reported latencies (excluding network overhead!)
                    latency = data.get("latency", {})
                    if latency:
                        embed_ms.append(latency.get("embedding_ms", 0))
                        retr_ms.append(latency.get("retrieval_ms", 0))
                        extr_ms.append(latency.get("generation_ms", 0) if latency.get("generation_ms") else latency.get("extractive_ms", 0))
                        ground_ms.append(latency.get("grounding_ms", 0))
                        core_ms.append(latency.get("rag_core_ms", 0))
                else:
                    print(f"Error on query {i}: {resp.status_code}")
            except Exception as e:
                print(f"Exception on query {i}: {e}")
                
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(queries)}")

    def stats(arr):
        if not arr: return {"p50": 0.0, "p70": 0.0, "p100": 0.0, "mean": 0.0}
        a = np.array(arr)
        return {
            "p50": float(np.percentile(a, 50)),
            "p70": float(np.percentile(a, 70)),
            "p100": float(np.percentile(a, 100)),
            "mean": float(np.mean(a)),
        }
        
    print("\n\nMANGOVOICE LIVE API BENCHMARK")
    print("=" * 60)
    print(f"Queries: {len(core_ms)}")
    print(f"Answer Rate: {answered}/{len(core_ms)}")
    print("\nStage                P50     P70     P100    Mean")
    print("-" * 60)
    
    e = stats(embed_ms)
    r = stats(retr_ms)
    x = stats(extr_ms)
    g = stats(ground_ms)
    c = stats(core_ms)
    
    print(f"Embedding            {e['p50']:>5.1f}   {e['p70']:>5.1f}   {e['p100']:>5.1f}   {e['mean']:>5.1f}")
    print(f"Retrieval            {r['p50']:>5.1f}   {r['p70']:>5.1f}   {r['p100']:>5.1f}   {r['mean']:>5.1f}")
    print(f"Extractive/Gen       {x['p50']:>5.1f}   {x['p70']:>5.1f}   {x['p100']:>5.1f}   {x['mean']:>5.1f}")
    print(f"Grounding            {g['p50']:>5.1f}   {g['p70']:>5.1f}   {g['p100']:>5.1f}   {g['mean']:>5.1f}")
    print("-" * 60)
    print(f"RAG CORE TOTAL       {c['p50']:>5.1f}   {c['p70']:>5.1f}   {c['p100']:>5.1f}   {c['mean']:>5.1f}")

if __name__ == "__main__":
    asyncio.run(run_live_benchmark())
