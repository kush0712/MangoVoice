"""
MangoVoice — FastAPI entry point for Vercel Python runtime.
Vercel expects the ASGI app exported as `app` from api/index.py.
"""
import sys
import os
import json
import time
from pathlib import Path

# Ensure backend package is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
import orjson
import numpy as np

from backend.config import settings
from backend.schemas import QueryResponse, HealthResponse, ErrorResponse
from backend.orchestrator import orchestrate_query, get_polished_result
from backend.telemetry import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="MangoVoice API",
    description="Voice-enabled RAG pipeline over MSMARCO-XI",
    version="1.0.0",
    default_response_class=ORJSONResponse,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def on_startup():
    """Warm up models and indices into RAM so the first real request is fast."""
    logger.info("Warming up FastEmbed model and LanceDB indices...")
    import asyncio
    from backend.retrieval.embeddings import get_embedder
    from backend.retrieval.lancedb_store import get_store
    from backend.retrieval.retriever import _RETRIEVAL_EXECUTOR

    # 1. Load FastEmbed ONNX model into RAM (one-time cost: ~2s)
    embedder = get_embedder()

    # 2. Trigger LanceDB open + vector matrix + BM25 scipy index into RAM
    #    (one-time cost: ~5-6s for 63k docs). After this every retrieval
    #    call uses in-RAM numpy/scipy with zero disk IO.
    store = get_store()
    if store.is_ready():
        loop = asyncio.get_running_loop()
        dummy_vec = np.zeros(384, dtype=np.float32)
        try:
            # Fire one dummy search to force any lazy initialisation
            await loop.run_in_executor(_RETRIEVAL_EXECUTOR, store.dense_search, dummy_vec, 1)
            await loop.run_in_executor(_RETRIEVAL_EXECUTOR, store.bm25_search, "warmup query", 1)
            logger.info("Warmup complete — model and indices loaded into RAM.")
        except Exception as e:
            logger.warning("Warmup query failed: %s", e)

@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Backend warm-up + readiness check."""
    from backend.retrieval.lancedb_store import get_store
    from backend.retrieval.embeddings import get_embedder

    try:
        store = get_store()
        embedder = get_embedder()
        index_ok = store.is_ready()
        embedder_ok = embedder.is_ready()
        numpy_ok = store._vec_normed is not None
        n_vecs = int(store._vec_normed.shape[0]) if numpy_ok else 0
    except Exception as exc:
        logger.warning("Health check degraded: %s", exc)
        index_ok = False
        embedder_ok = False
        numpy_ok = False
        n_vecs = 0

    # Log so we can confirm from Railway logs whether numpy index loaded
    logger.info(
        "Health: index=%s embedder=%s numpy_in_ram=%s vecs=%d",
        index_ok, embedder_ok, numpy_ok, n_vecs,
    )
    return HealthResponse(
        status="ready" if (index_ok and embedder_ok) else "degraded",
        index_version=settings.index_version,
        embedding_model=settings.embedding_model,
        index_ready=index_ok,
        embedder_ready=embedder_ok,
        numpy_in_ram=numpy_ok,
    )


@app.post("/api/query/text", response_model=QueryResponse)
async def query_text(body: dict) -> QueryResponse:
    """
    Text-only query endpoint (for demo queries / testing without microphone).
    Body: {"text": "...", "language": "auto"}
    """
    text = (body.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty query text")
    if len(text) > settings.max_transcript_chars:
        raise HTTPException(status_code=400, detail="Query too long")

    result = await orchestrate_query(transcript_text=text, language=body.get("language", "auto"))
    return result


@app.get("/api/query/result/{request_id}")
async def get_polished_answer(request_id: str) -> dict:
    """
    Progressive enhancement polling endpoint.

    The extractive fast-path answer is returned immediately in <200ms.
    Groq generates a polished, LLM-verified answer in the background (~1-2s).
    Frontend polls this endpoint once after ~1.5s to swap in the AI-enhanced
    answer if it passed grounding verification.

    Returns:
      {"ready": true,  "answer": "...", "answer_source": "llm", "grounding_score": 0.82}
      {"ready": false} — still generating or grounding failed
    """
    polished = get_polished_result(request_id)
    if polished is None:
        return {"ready": False}
    return {
        "ready": True,
        "answer": polished["answer"],
        "answer_source": polished["answer_source"],
        "grounding_score": polished["grounding_score"],
        "cited_chunk_ids": polished.get("cited_chunk_ids", []),
    }


@app.post("/api/tts")
async def text_to_speech(body: dict):
    """
    Generate authentic Indic voice audio via Sarvam Bulbul v2.
    Body: {"text": "...", "language": "hi-IN" | "en-IN" | "auto"}
    """
    import httpx
    text = (body.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text")

    if not settings.has_sarvam_key:
        raise HTTPException(status_code=503, detail="Sarvam API key not configured")

    # Detect language if auto
    lang = body.get("language", "auto")
    if lang == "auto":
        has_devanagari = any("\u0900" <= ch <= "\u097f" for ch in text)
        target_lang = "hi-IN" if has_devanagari else "en-IN"
    else:
        target_lang = "hi-IN" if "hi" in lang.lower() else "en-IN"

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.post(
                "https://api.sarvam.ai/text-to-speech",
                headers={
                    "api-subscription-key": settings.sarvam_api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "inputs": [text[:500]],
                    "target_language_code": target_lang,
                    "speaker": "anushka",
                    "model": "bulbul:v2",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            audios = data.get("audios", [])
            if audios:
                return {"audio_base64": audios[0], "format": "wav", "language": target_lang}
            raise HTTPException(status_code=500, detail="No audio returned from Sarvam")
        except Exception as exc:
            logger.warning("Sarvam TTS error: %s", exc)
            raise HTTPException(status_code=502, detail=str(exc))


# ── Hardcoded multilingual test set for /api/benchmark ───────────────────────
# 50 queries across English, Hindi, Hinglish — representative of real traffic.
# Self-contained so the endpoint works on Railway without val_passages.jsonl.
# N=50 is fast enough for free-tier Railway (all embedding lookups are LRU-cached
# after the first pass, so the benchmark completes in ~3-5s warm).
_BENCHMARK_QUERIES = [
    # English — factual domain queries (20)
    "What is the capital of France?",
    "How does photosynthesis work?",
    "What causes earthquakes?",
    "Who invented the telephone?",
    "What is the speed of light?",
    "How many bones are in the human body?",
    "What is the boiling point of water?",
    "Who wrote Romeo and Juliet?",
    "What are the symptoms of malaria?",
    "When did World War 2 end?",
    "What is GDP and how is it measured?",
    "How does the immune system fight viruses?",
    "What is the distance from Earth to the Sun?",
    "Who was the first person to walk on the moon?",
    "What is the chemical formula for water?",
    "How does a vaccine work?",
    "What is the largest planet in the solar system?",
    "Who painted the Mona Lisa?",
    "What causes thunder and lightning?",
    "How does the human heart work?",
    # Hindi — Devanagari queries (15)
    "\u092d\u093e\u0930\u0924 \u0915\u0940 \u0930\u093e\u091c\u0927\u093e\u0928\u0940 \u0915\u094d\u092f\u093e \u0939\u0948?",
    "\u0938\u0942\u0930\u094d\u092f \u0938\u0947 \u092a\u0943\u0925\u094d\u0935\u0940 \u0915\u0940 \u0926\u0942\u0930\u0940 \u0915\u093f\u0924\u0928\u0940 \u0939\u0948?",
    "\u092e\u0939\u093e\u0924\u094d\u092e\u093e \u0917\u093e\u0902\u0927\u0940 \u0915\u093e \u091c\u0928\u094d\u092e \u0915\u092c \u0939\u0941\u0906 \u0925\u093e?",
    "\u092d\u093e\u0930\u0924 \u092e\u0947\u0902 \u0915\u093f\u0924\u0928\u0947 \u0930\u093e\u091c\u094d\u092f \u0939\u0948\u0902?",
    "\u092a\u093e\u0928\u0940 \u0915\u093e \u0930\u093e\u0938\u093e\u092f\u0928\u093f\u0915 \u0938\u0942\u0924\u094d\u0930 \u0915\u094d\u092f\u093e \u0939\u0948?",
    "\u092a\u094d\u0930\u0915\u093e\u0936 \u0938\u0902\u0936\u094d\u0932\u0947\u0937\u0923 \u0915\u094d\u092f\u093e \u0939\u094b\u0924\u093e \u0939\u0948?",
    "\u092e\u0932\u0947\u0930\u093f\u092f\u093e \u0915\u0947 \u0932\u0915\u094d\u0937\u0923 \u0915\u094d\u092f\u093e \u0939\u0948\u0902?",
    "\u0921\u0940\u090f\u0928\u090f \u0915\u094d\u092f\u093e \u0939\u094b\u0924\u093e \u0939\u0948?",
    "\u092e\u093e\u0928\u0935 \u0936\u0930\u0940\u0930 \u092e\u0947\u0902 \u0915\u093f\u0924\u0928\u0940 \u0939\u0921\u094d\u0921\u093f\u092f\u093e\u0902 \u0939\u094b\u0924\u0940 \u0939\u0948\u0902?",
    "\u092a\u0943\u0925\u094d\u0935\u0940 \u0915\u093e \u0938\u092c\u0938\u0947 \u092c\u095c\u093e \u0917\u094d\u0930\u0939 \u0915\u094c\u0928 \u0938\u093e \u0939\u0948?",
    "\u0932\u0947\u091c\u0930 \u0915\u094d\u092f\u093e \u0939\u094b\u0924\u093e \u0939\u0948?",
    "\u0935\u093f\u0915\u093e\u0938 \u0926\u0930 \u0915\u094d\u092f\u093e \u0939\u094b\u0924\u0940 \u0939\u0948?",
    "\u090a\u0930\u094d\u091c\u093e \u0938\u0902\u0930\u0915\u094d\u0937\u0923 \u0915\u094d\u092f\u094b\u0902 \u091c\u0930\u0942\u0930\u0940 \u0939\u0948?",
    "\u092d\u0942\u0915\u0902\u092a \u0915\u094d\u092f\u094b\u0902 \u0906\u0924\u093e \u0939\u0948?",
    "\u0907\u0902\u091f\u0930\u0928\u0947\u091f \u0915\u0948\u0938\u0947 \u0915\u093e\u092e \u0915\u0930\u0924\u093e \u0939\u0948?",
    # Hinglish — code-mixed queries (15)
    "Blood pressure normal range kya hota hai?",
    "Diabetes ke symptoms kya hain?",
    "India mein kitni official languages hain?",
    "Oxygen ka atomic number kya hai?",
    "DNA kya hota hai aur kaise kaam karta hai?",
    "Earthquake kyu aata hai?",
    "Vaccine kaise kaam karta hai body mein?",
    "Solar system mein kitne planets hain?",
    "World War 2 kab khatam hua?",
    "GDP kya hota hai?",
    "Immune system virus ko kaise fight karta hai?",
    "Moon par pehle kaun gaya tha?",
    "Mona Lisa kisne banayi thi?",
    "Photosynthesis kya hota hai?",
    "Human heart kaise kaam karta hai?",
]


@app.get("/api/benchmark")
async def live_benchmark(
    n: int = Query(default=20, ge=5, le=100, description="Number of queries to run (5–100)"),
) -> dict:
    """
    Live benchmark — honest per-query latency measurement.

    Each query is run with a FRESH embedding call (bypass LRU cache) and
    bypasses the retrieval result cache so you see real compute time, not
    dict-lookup time. This shows what a first-ever query for each question
    actually costs on this hardware.

    After the first call, repeated identical queries are served from the
    in-process LRU cache (~0ms embedding, ~0ms retrieval) for subsequent
    users — not reported here to avoid misleading numbers.

    Example: GET /api/benchmark?n=20
    """
    from backend.retrieval.lancedb_store import get_store
    from backend.retrieval.embeddings import get_embedder
    from backend.retrieval.fusion import rrf_fuse
    from backend.retrieval.confidence import compute_retrieval_confidence, should_generate
    from backend.guardrails import normalize_text, layer1_check
    from backend.fallback.extractive import extractive_fallback
    from backend.grounding import verify_grounding_extractive

    store = get_store()
    embedder = get_embedder()

    queries = list(_BENCHMARK_QUERIES)
    if n > len(queries):
        candidates: list[str] = []
        for path_cand in [
            Path("data/val_passages.jsonl"),
            Path(__file__).resolve().parent.parent / "data" / "val_passages.jsonl",
            Path("/app/data/val_passages.jsonl"),
        ]:
            if path_cand.exists():
                try:
                    with open(path_cand, "r", encoding="utf-8") as f:
                        for line in f:
                            if len(queries) + len(candidates) >= n:
                                break
                            line_str = line.strip()
                            if not line_str:
                                continue
                            row = json.loads(line_str)
                            q = row.get("query", "")
                            if q and len(q) > 5 and q not in queries and q not in candidates:
                                candidates.append(q)
                except Exception as e:
                    logger.warning("Failed to load additional benchmark queries: %s", e)
                break
        queries.extend(candidates)

    queries = queries[:n]
    embed_ms_list, retr_ms_list, extr_ms_list, ground_ms_list, core_ms_list = [], [], [], [], []
    statuses = []

    for query in queries:
        t_core = time.perf_counter()

        normalized = normalize_text(query)
        l1 = layer1_check(normalized)
        if not l1.passed:
            statuses.append("refused_l1")
            core_ms_list.append((time.perf_counter() - t_core) * 1000)
            embed_ms_list.append(0.0)
            retr_ms_list.append(0.0)
            extr_ms_list.append(0.0)
            ground_ms_list.append(0.0)
            continue

        # ── Embed: fresh call every time (bypass LRU to measure real ONNX cost)
        t0 = time.perf_counter()
        import asyncio, concurrent.futures
        loop = asyncio.get_running_loop()
        query_vec = await loop.run_in_executor(
            None, lambda: embedder.embed_one(normalized)
        )
        embedding_ms = (time.perf_counter() - t0) * 1000

        # ── Retrieval: call store directly, bypassing retrieval result cache
        t0 = time.perf_counter()
        dense_results = store.dense_search(query_vec, settings.dense_top_k)
        bm25_results = store.bm25_search(normalized, settings.bm25_top_k)
        fused = rrf_fuse(dense_results, bm25_results, top_k=settings.final_top_k)
        retrieval_result = compute_retrieval_confidence(
            sources=fused,
            dense_sources=dense_results,
            bm25_sources=bm25_results,
        )
        retrieval_ms = (time.perf_counter() - t0) * 1000

        if not should_generate(retrieval_result):
            statuses.append("refused_low_confidence")
            extr_ms = 0.0
            ground_ms = 0.0
        else:
            sources = retrieval_result.sources

            t0 = time.perf_counter()
            fast_answer = extractive_fallback(sources, normalized, reason="benchmark")
            extr_ms = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            grounding = verify_grounding_extractive(fast_answer, sources)
            ground_ms = (time.perf_counter() - t0) * 1000

            statuses.append("answered" if grounding.passed else "refused_grounding")

        core_ms = (time.perf_counter() - t_core) * 1000
        embed_ms_list.append(embedding_ms)
        retr_ms_list.append(retrieval_ms)
        extr_ms_list.append(extr_ms)
        ground_ms_list.append(ground_ms)
        core_ms_list.append(core_ms)

    def _stats(arr):
        if not arr:
            return {
                "p50_ms": 0.0,
                "p70_ms": 0.0,
                "p90_ms": 0.0,
                "p99_ms": 0.0,
                "p100_ms": 0.0,
                "mean_ms": 0.0,
            }
        a = np.array(arr)
        return {
            "p50_ms": round(float(np.percentile(a, 50)), 2),
            "p70_ms": round(float(np.percentile(a, 70)), 2),
            "p90_ms": round(float(np.percentile(a, 90)), 2),
            "p99_ms": round(float(np.percentile(a, 99)), 2),
            "p100_ms": round(float(np.percentile(a, 100)), 2),
            "mean_ms": round(float(np.mean(a)), 2),
        }

    answered = statuses.count("answered")
    return {
        "benchmark": "A — Fast-path RAG (extractive, no LLM)",
        "description": (
            "normalize → guardrails → embed (fresh ONNX, no cache) → "
            "retrieve (RAM: numpy dense + numpy BM25, no cache) → "
            "extractive → grounding_extractive → response. "
            "Groq excluded (no free-tier quota consumed). "
            "Embedding and retrieval caches BYPASSED for honest measurement."
        ),
        "n_queries": len(queries),
        "query_languages": "English, Hindi, Hinglish",
        "answer_rate": f"{answered}/{len(queries)}",
        "stages": {
            "embedding_fastembed": _stats(embed_ms_list),
            "retrieval_lancedb_hybrid": _stats(retr_ms_list),
            "extractive_answer": _stats(extr_ms_list),
            "grounding_extractive": _stats(ground_ms_list),
        },
        "rag_core_total": _stats(core_ms_list),
        "sla_target_ms": 200,
        "sla_met": bool(np.percentile(core_ms_list, 70) <= 200) if core_ms_list else False,
        "note_cache_benefit": (
            "Repeated queries (same question asked again) are served from "
            "process-local LRU cache: embedding ~0ms, retrieval ~0ms. "
            "This benchmark deliberately bypasses caches to show worst-case "
            "(first-ever query) latency."
        ),
        "note_benchmark_b": (
            "LLM-enhanced (Groq) latency: ~700-1500ms P50. "
            "Run: python -m evaluation.latency_benchmark --mode b"
        ),
        "note_benchmark_c": (
            "Voice E2E adds Sarvam STT (~300-800ms network round-trip), "
            "always reported separately in LatencyMetrics.stt_ms."
        ),
    }
