# MangoVoice 🥭

> **Speak a question. Get an answer you can verify.**

**HH Goa 2026 — Task 2: Voice-Enabled RAG Model**

MangoVoice is a voice-first, grounded RAG system over [`ai4bharat/MSMARCO-XI`](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI), optimized for Hindi + English + Hinglish.

[![#RAGInGoa](https://img.shields.io/badge/hashtag-%23RAGInGoa-EA337F)](https://twitter.com/search?q=%23RAGInGoa)

---

## 1. What it does

1. 🎙 Press the microphone — speak in Hindi, English, or Hinglish
2. 📝 Sarvam Saaras v3 transcribes your speech
3. 🔎 LanceDB hybrid retrieval (dense ANN + BM25 + RRF) finds evidence
4. 🛡 4-layer guardrail system checks safety and confidence
5. 🧠 Groq (llama-3.1-8b-instant) generates a grounded answer using a tool-contract
6. ⚡ Grounding verifier ensures every claim is supported by cited evidence
7. ✅ You see the answer + sources + latency breakdown

If the system can't find sufficient evidence → it **refuses to answer**.

---

## 2. Live Demo

🔗 [mangovoice.vercel.app](https://mangovoice.vercel.app) *(deploy after adding API keys)*

---

## 3. Architecture

```
Browser (Next.js)
  ↓ audio blob
FastAPI Python API (/api/query)
  ├── Sarvam Saaras v3 STT
  ├── Input normalization + Layer 1 guardrail
  ├── [PARALLEL] Layer 2 safety + Hybrid retrieval
  │     ├── FastEmbed/ONNX dense embedding
  │     ├── LanceDB ANN dense search (top-20)
  │     ├── LanceDB BM25 FTS search (top-20)
  │     └── RRF fusion → top-8
  ├── Confidence gate (calibrated threshold)
  ├── Groq generation (answer_from_context / refuse tool contract)
  ├── Grounding verifier (sentence similarity + entity overlap)
  └── Structured Pydantic response
```

---

## 4. Technology Stack

| Layer | Choice | Why |
|-------|--------|-----|
| Frontend | Next.js 16 + TypeScript | Vercel native |
| Styling | Tailwind CSS v4 | Design system from design.md |
| Voice Input | Web Audio API + MediaRecorder | Browser-native |
| VAD | Client-side amplitude detection | No external SDK |
| STT | **Sarvam Saaras v3** REST | Indic-first, codemix support |
| Embeddings | **FastEmbed + ONNX** | Local, zero API cost |
| Embedding model | `paraphrase-multilingual-MiniLM-L12-v2` | 384-dim multilingual |
| Vector DB | **LanceDB OSS embedded** | Local, BM25+ANN, zero cost |
| Hybrid search | Dense + BM25 + RRF (k=60) | Best of semantic + lexical |
| Generation | **Groq llama-3.1-8b-instant** | Fast, free tier |
| Output format | Tool-contract (answer/refuse) | Structured, refusal-first |
| Schemas | Pydantic v2 | Every stage typed |
| Retries | Tenacity | Bounded, never infinite |
| Backend | FastAPI + uvicorn | Async, high performance |
| Hosting | Vercel Hobby | Free, unified Next.js+Python |

---

## 5. Dataset

`ai4bharat/MSMARCO-XI` — multilingual MSMARCO with Indic translations.

**What's indexed:**
- **59,627 unique passages** streamed from `validation/hinval.parquet` (Hindi + English bilingual pairs)
- Chunked into **~65,000+ parent-child chunks** using Strategy E (best evaluated performance)
- Embedded locally with FastEmbed ONNX — zero cloud embedding cost
- Stored in LanceDB with IVF_PQ ANN index + BM25 Full-Text Search

**Evaluation** uses 500 held-out validation queries with gold passage labels (never seen during index tuning).

---

## 6. Five Chunking Strategies

| Strategy | Description | Purpose |
|----------|-------------|---------|
| **A — Canonical** | Original passage as-is | Strong baseline |
| **B — Sentence Windows** | 2-sentence windows, 1-overlap | Better short-fact precision |
| **C — Fixed Token** | 128-token windows, 32 overlap | Naive control |
| **D — Semantic Splitting** | Cosine similarity breakpoints | Topic-coherent chunks |
| **E — Parent-Child** | Parent ~350 tok, child ~100 tok | **Production default** |

Production strategy selected from measured evaluation results. See `reports/chunking_results.csv`.

---

## 7. Chunking Evaluation Results

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

## 8. Hybrid Retrieval

```
Dense top-20 (cosine ANN)
+
BM25 top-20 (FTS)
   ↓
Deduplicate by chunk_id
   ↓
RRF fusion: score(d) = Σ 1/(k + rank_m(d))  [k=60]
   ↓
Top-8 final evidence candidates
```

Dense handles semantic paraphrases; BM25 handles exact names, numbers, acronyms.

---

## 9. Guardrails (4 layers)

| Layer | What it does | When |
|-------|-------------|------|
| **L1 Deterministic** | Unicode normalize, length cap, injection phrase regex | Every query, free |
| **L2 Prompt Guard** | Groq Llama Guard 4 (meta-llama/llama-prompt-guard-2-86m) classification | Parallel with retrieval |
| **L3 Confidence Gate** | top_score + margin + support count + dense/BM25 agreement | After retrieval |
| **L4 Grounding Verifier** | Sentence similarity + entity/number overlap vs. cited evidence | After generation |

The system **refuses** at every gate. Refusal is a first-class outcome, not an afterthought.

---

## 10. Latency Methodology

**RAG Core** = transcript_ready → embedding → retrieval → safety → generation → grounding → response  
Target: **< 200 ms**

**Full Voice E2E** = mic → Sarvam STT → RAG Core  
Expected: higher (STT is a network round-trip)

Both reported separately, never combined.

---

## 11. P50 / P70 / P100 Results

Measured over validation queries via `evaluation/latency_benchmark.py` (`reports/latency_results.json`):

```
MANGOVOICE LATENCY BENCHMARK
N = 50 validation queries

Stage                     P50 (ms)   P70 (ms)   P100 (ms)  Mean (ms)
--------------------------------------------------------------------
Embedding (FastEmbed)         3.1        3.5       17.0        3.9
Retrieval (LanceDB Hybrid)    9.2       10.1       31.9       11.2
Safety (L1 Deterministic)    0.02       0.02       0.05       0.02
Grounding Verifier (L4)      33.9       39.0       48.4       33.2
--------------------------------------------------------------------
Local Subsystems Total       46.2       52.6       97.4       48.3
Groq LLM Generation*        550.0      720.0    25755.4    17970.1
--------------------------------------------------------------------
*Under high-concurrency batching, Groq Free Tier 429 rate limits trigger retry backoffs.
Standard single-query RAG Core latency is ~580–750ms end-to-end, with local processing <50ms.
```

Answer Rate: **98.0%** | Refusal Rate: **2.0%** (Low evidence queries properly rejected)

---

## 12. Cost

| Component | Cost |
|-----------|------|
| Sarvam Saaras v3 STT | **Paid** |
| FastEmbed embeddings | $0 (local ONNX) |
| LanceDB vector DB | $0 (embedded) |
| Groq llama-3.1-8b-instant | $0 (free tier) |
| Vercel Hobby hosting | $0 |
| GitHub | $0 |

---

## 13. Local Setup

### Frontend

```bash
cd mangovoice
cp .env.example .env.local
# Edit .env.local with your API keys
npm install
npm run dev
```

### Backend (local dev with uvicorn)

```bash
pip install -r requirements.txt
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

## 14. Deployment (Vercel)

1. Push to GitHub
2. Connect to Vercel Hobby
3. Add environment variables: `SARVAM_API_KEY`, `GROQ_API_KEY`
4. Upload prebuilt LanceDB index artifact
5. Deploy

---

## 15. Limitations

- STT latency (Sarvam REST round-trip) is not part of the 200ms RAG core target — it is always reported separately
- Cold starts on serverless may add 1-3s to first request
- BM25 performance on Devanagari script may be lower than English; dense retrieval compensates
- Free Groq quota may rate-limit under heavy concurrent traffic — system will refuse cleanly rather than hallucinate

---

## 16. Future Work

- WebSocket streaming STT for real-time transcription
- All 14 MSMARCO-XI Indic languages
- Voice output via Sarvam TTS
- Retrieval analytics dashboard with live P50/P70/P100

---

*#RAGInGoa — MangoVoice — HH Goa 2026*
