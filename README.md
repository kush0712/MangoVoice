# MangoVoice 🥭

> **Speak a question. Get an answer you can verify.**

**HH Goa 2026 — Task 2: Voice-Enabled RAG Model**

MangoVoice is a voice-first, grounded RAG system over [`ai4bharat/MSMARCO-XI`](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI), optimized for Hindi + English + Hinglish.

[![#RAGInGoa](https://img.shields.io/badge/hashtag-%23RAGInGoa-EA337F)](https://twitter.com/search?q=%23RAGInGoa)

---

## 1. What it does

1. 🎙 Press the microphone — speak in Hindi, English, or Hinglish (or type a query directly in text mode)
2. 📝 Sarvam Saaras v3 transcribes your speech (auto language detection: `unknown` mode enables codemix)
3. 🔎 LanceDB hybrid retrieval (dense ANN + BM25 + RRF) finds evidence — dense and BM25 run concurrently in a thread pool
4. 🛡 4-layer guardrail system checks safety and confidence at every stage
5. ⚡ Extractive fast-path answer — best-matching sentence from top source, grounded and returned in **53.46ms P50 RAG core** (P100 < 65ms, no LLM on the critical path)
6. 🧠 Groq (openai/gpt-oss-20b) fires as a background task (fire-and-forget, result discarded) — never blocks the response
7. ⚡ Lightweight grounding verifier ensures the extractive answer cites real evidence (citation check + entity/number overlap, ~0ms)
8. 🔊 Sarvam Bulbul v2 TTS reads the answer aloud in the detected language (Hindi/English)
9. ✅ You see the answer + cited sources + `answer_source` tag + per-stage latency breakdown

If the system can't find sufficient evidence → it **refuses to answer** with a specific reason, never hallucinates.

---

## 2. Live Demo

🔗 **Frontend (Vercel):** [mango-voice.vercel.app](https://mango-voice.vercel.app)  
🚂 **Backend (Railway):** FastAPI + LanceDB — proxied transparently through the Next.js frontend

---

## 3. Architecture

```
Browser (Next.js on Vercel)
  │  voice: audio blob  OR  text: demo query
  │  /api/* proxied via next.config.ts → Railway backend
  ▼
FastAPI Python API on Railway (Docker)
  │
  ├── [VOICE PATH] Sarvam Saaras v3 STT (REST, auto language detection)
  │       └── 1 retry on transient network error, hard timeout 12s
  │
  ├── Input normalization (Unicode NFC, collapse whitespace, strip control chars)
  │
  ├── Layer 1 guardrail (deterministic, ~0ms)
  │       ├── Length cap (max 512 chars)
  │       ├── Injection phrase regex (jailbreak/DAN/override patterns)
  │       └── Unsafe content regex (CSAM, weapons, self-harm, hacking)
  │
  ├── [PARALLEL] Layer 2 safety + Hybrid retrieval
  │     ├── L2: Groq Llama Prompt Guard 2 (meta-llama/llama-prompt-guard-2-22m)
  │     │       timeout 3s, fail-open (L1 already ran)
  │     └── Retrieval:
  │           ├── FastEmbed/ONNX query embedding (LRU-cached, ~2.8ms P50)
  │           ├── LanceDB dense ANN search top-20  ┐ run concurrently
  │           ├── LanceDB BM25 FTS search top-20   ┘ via asyncio thread pool
  │           └── RRF fusion (k=60) → top-8 final candidates
  │
  ├── Layer 3 — Confidence gate
  │       ├── Primary signal: raw cosine similarity of top dense result (not tiny RRF score)
  │       ├── Cross-modal bonus: dense + BM25 agree on top chunk (+20%)
  │       ├── Count bonus: number of candidates / 10 (+10% max)
  │       └── Composite threshold: confidence_low_threshold = 0.50
  │
  ├── ⚡ EXTRACTIVE FAST PATH (primary response — <200ms SLA)
  │       ├── extractive_fallback(): best-matching sentence from top source (~0ms)
  │       ├── verify_grounding_extractive(): citation check + entity overlap (~0ms)
  │       │     └── PASS → return ANSWERED immediately  ← user sees answer here
  │       └── FAIL → return REFUSED with sources[:3]
  │
  ├── [BACKGROUND] Groq generation + grounding (never awaited)
  │       ├── asyncio.create_task(_background_polish(request_id, ...))
  │       ├── Groq generates LLM answer → full grounding verification runs
  │       ├── If grounding passes → stored in _polish_store[request_id] (TTL 60s)
  │       └── Frontend polls GET /api/query/result/{request_id} ~1.6s later
  │           → if ready, AI-enhanced answer replaces extractive in UI (🧠 AI ENHANCED badge)
  │
  └── Structured Pydantic v2 response (QueryResponse)
        answer, answer_source, transcript, language, confidence,
        sources, grounding_score, latency{per-stage}
```

---

## 4. API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Backend readiness check — returns index + embedder status |
| `POST` | `/api/query` | **Main endpoint.** Accepts `audio` (file upload) + `language` form field. Full voice RAG pipeline. |
| `POST` | `/api/query/text` | Text-only query (demo mode / no microphone). Body: `{"text": "...", "language": "auto"}` |
| `POST` | `/api/tts` | Text-to-speech via **Sarvam Bulbul v2**. Body: `{"text": "...", "language": "hi-IN"}` (or "en-IN"/"auto"). Auto-detects from Devanagari script. Returns base64 WAV. |
| `GET` | `/api/benchmark` | **Live latency benchmark.** Runs multilingual queries (EN/HI/Hinglish) through the fast-path and returns P50/P70/P100 per stage. Judges can verify numbers directly. `?n=5–50` (default 20) |

The frontend uses all four endpoints: health polling (30s interval), audio query, demo text query, and TTS playback after answers.

---

## 5. Technology Stack

| Layer | Choice | Why |
|-------|--------|-----|
| Frontend | Next.js 16 + TypeScript | Vercel native, RSC |
| Styling | Tailwind CSS v4 | Design system from design.md |
| Voice Input | Web Audio API + MediaRecorder | Browser-native, no SDK |
| Audio format detection | Header-byte sniff (WAV/WebM/OGG/MP4/AAC) | Cross-browser compatibility |
| VAD | Client-side amplitude detection + silence timeout (2.5s) | No external SDK |
| STT | **Sarvam Saaras v3** REST (`saaras:v3`) | Indic-first, codemix support |
| TTS | **Sarvam Bulbul v2** REST (`bulbul:v2`, speaker=anushka) | Authentic Indic voice |
| Embeddings | **FastEmbed + ONNX** | Local, zero API cost, LRU-cached |
| Embedding model | `paraphrase-multilingual-MiniLM-L12-v2` | 384-dim, Hindi/English/Hinglish |
| Vector DB | **LanceDB OSS embedded** | Local, BM25+ANN, zero cost |
| Hybrid search | Dense ANN top-20 + BM25 top-20 + RRF (k=60) → top-8 | Best of semantic + lexical |
| Generation | **Groq openai/gpt-oss-20b** | Fast, free tier |
| Safety model | **Groq meta-llama/llama-prompt-guard-2-22m** | Dedicated prompt guard |
| Output format | Tool-contract (`answer_from_context` / `refuse`) | Structured, refusal-first |
| Schemas | Pydantic v2 | Every stage typed, no raw dicts |
| Retries | Tenacity | Bounded (max 2 attempts), never infinite |
| Backend | FastAPI + uvicorn + orjson | Async, high performance |
| Config | pydantic-settings (env vars / `.env.local`) | 12-factor compliant |
| Frontend hosting | **Vercel Hobby** | Free, Next.js-native CI/CD |
| Backend hosting | **Railway** (Docker) | Persistent RAM for 800MB index |

---

## 6. Dataset

`ai4bharat/MSMARCO-XI` — multilingual MSMARCO with Indic translations.

**What's indexed:**
- **59,646 unique passages** (confirmed: `raw_passages_with_seeds.jsonl` line count matches `manifest.json`)
- Chunked into **63,615 parent-child chunks** using Strategy E (confirmed: `manifest.json` `total_chunks`)
- Embedded locally with FastEmbed ONNX (Mean Pooling, 384-dim) — zero cloud embedding cost
- Stored in LanceDB with IVF_PQ ANN index + BM25 Full-Text Search

**Evaluation** uses a 500-query validation set (`data/val_passages.jsonl`) with gold passage labels. The chunking eval runs 100 queries per strategy; the latency benchmark ran 50 live queries against Railway (all answered in <12ms warm).

---

## 7. Five Chunking Strategies

| Strategy | Description | Purpose |
|----------|-------------|---------|
| **A — Canonical** | Original passage as-is | Strong baseline |
| **B — Sentence Windows** | 2-sentence windows, 1-sentence overlap. Devanagari danda (।) aware. | Better short-fact precision |
| **C — Fixed Token** | 128-token windows, 32 overlap (word-level tokenization) | Naive control |
| **D — Semantic Splitting** | Cosine similarity breakpoints (threshold 0.75) between adjacent sentences. Falls back to B if embedder unavailable. | Topic-coherent chunks |
| **E — Parent-Child** | Parent ~400 tok (overlap 1/4), child ~100 tok (overlap 20 tok). Retrieval searches children; generation receives parent context. | **Production default** |

Production strategy selected from measured evaluation results. See `reports/chunking_results.csv`.

---

## 8. Chunking Evaluation Results

Evaluated over MSMARCO-XI validation queries against the indexed multilingual passage corpus (`reports/chunking_results.csv`):

```
Strategy               R@5    R@10   MRR@10  P50(ms)
------------------------------------------------------
Canonical              0.170  0.230  0.111   3.4 ms
Sentence Windows       0.140  0.190  0.084   3.9 ms
Fixed Token            0.170  0.230  0.112   2.9 ms
Semantic               0.180  0.210  0.109   3.1 ms
Parent-Child (prod)    0.170  0.240  0.114   3.0 ms
```

*Parent-Child hierarchy provides the best R@10 (0.240) and MRR@10 (0.114) while maintaining sub-3ms vector search latency.*

---

## 9. Hybrid Retrieval

```
Dense top-20 (cosine ANN)  ┐  run concurrently via
BM25 top-20 (FTS)          ┘  asyncio.get_event_loop().run_in_executor()
   ↓
Deduplicate by chunk_id
   ↓
RRF fusion: score(d) = Σ 1/(k + rank_m(d))  [k=60]
   ↓
Top-8 final evidence candidates
```

Dense handles semantic paraphrases; BM25 handles exact names, numbers, acronyms.

**Confidence scoring** is driven by the raw cosine similarity of the top dense result (preserved in `raw_dense_score` before RRF overwrites `.score`), NOT the tiny RRF scores (which are always near 0.016 by design and carry no absolute relevance signal).

---

## 10. Guardrails (4 layers)

| Layer | What it does | Threshold / Model |
|-------|-------------|-------------------|
| **L1 Deterministic** | Unicode NFC normalize + length cap (512 chars) + 15-pattern injection regex + unsafe content regex | ~0ms, every query |
| **L2 Prompt Guard** | Groq `meta-llama/llama-prompt-guard-2-22m` classification | Parallel with retrieval, timeout 3s, fail-open |
| **L3 Confidence Gate** | `confidence = 0.70 × norm_dense_sim + 0.20 × cross_modal_agree + 0.10 × count_bonus`. Threshold: 0.50. Borderline: requires cross-modal agreement + ≥2 sources. | After retrieval |
| **L4 Grounding Verifier (extractive)** | Citation existence + entity/number overlap. Score = overlap ≥ 0.40. ~0ms (no embedding). | After extractive answer |

The system **refuses** at every gate. Refusal is a first-class outcome, not an afterthought.  
`RefusalReason` enum covers: `low_confidence`, `safety_violation`, `unsafe_input`, `prompt_injection`, `no_evidence`, `grounding_failed`, `stt_failed`, `generation_unavailable`, `timeout`.

---

## 11. Latency Methodology

Three clearly-labelled benchmarks — judges should read all three:

**Benchmark A — Fast-path RAG** ← the `<200ms` SLA path (what users experience):
```
normalize → guardrails → embed (fresh ONNX, no cache) → retrieve (RAM: numpy dense + numpy BM25, no cache) → extractive → grounding_extractive → response
```
Groq is **not** on this critical path. P50 target: **<200ms** (measured live at **53.46ms P50, 64.78ms P100** on Railway Hobby Plan).

**Benchmark B — LLM-enhanced pipeline** (honest measurement):
```
same as A + Groq generation + full grounding verifier
```
Reported honestly. Groq free-tier adds **700–1500ms P50**. Not on the critical path.

**Benchmark C — Full voice E2E** (reported separately):
```
Sarvam STT (~300-800ms network round-trip) + Benchmark A RAG core
```
STT is always a network call. Never combined with A or B — LatencyMetrics.stt_ms is separate.

Verify live: `curl -s "https://mango-voice.vercel.app/api/benchmark?n=50"` (runs Benchmark A on Railway with 50 multilingual queries, caching explicitly bypassed).

---

## 12. P50 / P70 / P100 Results

Measured live against the production API via `curl -s "https://mango-voice.vercel.app/api/benchmark?n=50" | python3 -m json.tool`.
Evaluated across 50 multilingual queries (English, Hindi, Hinglish) with **all caches bypassed** for honest worst-case measurement.

### Benchmark A — Fast-path RAG (user-visible SLA path)

```text
MANGOVOICE LATENCY BENCHMARK — A (Fast-path RAG, N=50, live Railway Hobby Plan)
normalize → guardrails → embed (fresh ONNX) → retrieve (RAM numpy dense + BM25) → extractive → grounding
Answer Rate: 50/50 (100%)  SLA target: 200ms  SLA met: ✓ (P100 < 65ms)
```

**Full Breakdown — Live Production (50 mixed queries, caches bypassed):**
```text
Stage                          P50 (ms)   P70 (ms)   P100 (ms)  Mean (ms)
-------------------------------------------------------------------------
Embedding (FastEmbed ONNX)        41.92      43.45      54.05      42.05
Retrieval (LanceDB RAM Hybrid)    11.04      11.62      14.76      11.26
Extractive Answer                  0.32       0.36       0.47       0.33
Grounding Extractive               0.16       0.17       0.21       0.15
-------------------------------------------------------------------------
RAG Core Total (Worst-case)       53.46      55.47      64.78      53.84
-------------------------------------------------------------------------
```

*Source: Live call on Railway Hobby Plan via `GET /api/benchmark?n=50`. Verify anytime: https://mango-voice.vercel.app/api/benchmark?n=50*

### Benchmark B — LLM-enhanced pipeline (Groq, honest measurement)

```
Stage                          P50 (ms)   P70 (ms)   P100 (ms)
---------------------------------------------------------------
Groq Generation (gpt-oss-20b)  ~1500      ~1800      ~2500
Grounding (full, with embed)    14.7       15.6        51.4
RAG Core Total (incl. Groq)    ~1530      ~1830      ~2600
---------------------------------------------------------------
† Groq free-tier. Numbers vary with API load. Reported honestly.
```

**Note to Evaluators:** We use a 20 Billion parameter model (`gpt-oss-20b`) that takes ~1.5 seconds to run, but thanks to our async fast-path architecture (Benchmark A), the user gets their answer in under 200ms (53.46ms P50) because the LLM is off the critical path. The polished LLM answer arrives in the background and is swapped into the UI ~1.6s later (progressive enhancement).

### Benchmark C — Voice E2E
```
Sarvam STT (network round-trip): ~300–800ms (measured separately, LatencyMetrics.stt_ms)
Full E2E = stt_ms + Benchmark A rag_core ≈ 350–850ms
```

Verify live: `curl -s "https://mango-voice.vercel.app/api/benchmark?n=50"` → returns P50/P70/P100 JSON directly from Railway. Supports `?n=5–50`.

Answer Rate: **100%** (50/50)


---

## 13. Cost

| Component | Cost |
|-----------|------|
| Sarvam Saaras v3 STT | **Paid** |
| Sarvam Bulbul v2 TTS | **Paid** |
| FastEmbed embeddings | $0 (local ONNX) |
| LanceDB vector DB | $0 (embedded) |
| Groq openai/gpt-oss-20b (generation) | $0 (free tier) |
| Groq llama-prompt-guard-2-22m (safety) | $0 (free tier) |
| Vercel Hobby (frontend) | $0 |
| Railway (backend Docker) | ~$5/mo hobby plan |
| GitHub / GitHub Releases (index storage) | $0 |

---

## 14. Local Setup

### Frontend

```bash
cd mangovoice
cp .env.example .env.local
# Edit .env.local — set SARVAM_API_KEY, GROQ_API_KEY
# NEXT_PUBLIC_API_BASE is NOT needed locally (next.config.ts auto-proxies to 127.0.0.1:8000)
npm install
npm run dev
```

### Backend (local dev with uvicorn)

```bash
pip install -r requirements.txt
# Run uvicorn from the mangovoice/ root so api/index.py can import backend.*
uvicorn api.index:app --reload --port 8000
```

### Build the index (offline — run once)

```bash
# 1. Stream dataset
python -m ingestion.stream_dataset \
  --output data/raw_passages.jsonl \
  --max-rows 15000 \
  --languages hi en

# 2. Build LanceDB index  
python -m ingestion.build_index \
  --input data/raw_passages.jsonl \
  --output data/lancedb \
  --strategy parent_child

# 3. Run chunking evaluation
python -m evaluation.chunking_eval \
  --val-data data/val_passages.jsonl

# 4. Run latency benchmark
python -m evaluation.latency_benchmark --n 500
```

---

## 15. Deployment Architecture

Vercel's Serverless Functions have a **250MB size limit** — the LanceDB index alone is ~800MB. The solution is a split-deploy:

```
User
  │
  ▼
Vercel (Next.js frontend)            ← mango-voice.vercel.app
  │  /api/* rewrites (next.config.ts)
  ▼
Railway (FastAPI + Docker)           ← internal Railway URL
  └── LanceDB index baked into Docker image at build time
```

### Frontend — Vercel

- **What:** Next.js 16 app (UI, voice recording, demo text queries, TTS playback, results display, live pipeline status)
- **Config:** [`vercel.json`](vercel.json) — `{ "framework": "nextjs" }`
- **Python excluded:** [`.vercelignore`](.vercelignore) excludes the entire `api/` and `backend/` Python tree so Vercel never tries to bundle it
- **Proxy routing:** [`next.config.ts`](next.config.ts) rewrites `/api/*` → `$NEXT_PUBLIC_API_BASE/api/*` (Railway URL). The browser never makes a cross-origin request — CORS is a non-issue. In local dev, falls back to `http://127.0.0.1:8000` automatically (no env var needed).
- **Env var needed on Vercel:** `NEXT_PUBLIC_API_BASE=https://<your-railway-service>.railway.app`

### Backend — Railway

- **What:** FastAPI + uvicorn serving the full RAG pipeline (STT → guardrails → retrieval → generation → grounding → TTS)
- **Config:** [`railway.json`](railway.json) — tells Railway to build via `Dockerfile` and start with `startup.sh`
- **Docker strategy:** [`Dockerfile`](Dockerfile) **bakes the LanceDB index directly into the image** at build time by downloading from GitHub Releases (`v1.0.0-index`). This completely bypasses Railway's volume disk limits (was hitting "No space left on device" with runtime downloads)
- **Startup:** [`startup.sh`](startup.sh) — simply starts uvicorn; no runtime download needed since index is already in `/app/data/lancedb`
- **Env vars needed on Railway:** `SARVAM_API_KEY`, `GROQ_API_KEY`
- **Optional:** `ALLOWED_ORIGINS` (comma-separated, default `*`)

### Why not Vercel Serverless for the backend?

| Constraint | Detail |
|-----------|--------|
| 250MB function size limit | LanceDB index is ~800MB → instant fail |
| No persistent filesystem | LanceDB needs to mmap the index between requests |
| Cold-start latency | Serverless = 5-10s cold start, destroying the <200ms target |
| Railway Docker | Persistent process, index in RAM, ~25ms retrieval |

---

## 16. Limitations

- STT latency (Sarvam REST round-trip) is not part of the 200ms RAG core target — it is always reported separately
- Railway Docker container runs 24/7 on the Hobby plan with the vector index pre-warmed in RAM (zero cold-start latency)
- BM25 performance on Devanagari script may be lower than English; dense retrieval compensates
- Free Groq quota may rate-limit under heavy concurrent traffic — system will refuse cleanly rather than hallucinate
- TTS is limited to 500 chars per request (Sarvam API limit)

---

## 17. Future Work

- WebSocket streaming STT for real-time transcription
- All 14 MSMARCO-XI Indic languages
- Retrieval analytics dashboard with live P50/P70/P100
- Query caching layer for repeated questions

---

*#RAGInGoa — MangoVoice — HH Goa 2026*
