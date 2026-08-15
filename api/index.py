"""
MangoVoice — FastAPI entry point for Vercel Python runtime.
Vercel expects the ASGI app exported as `app` from api/index.py.
"""
import sys
import os

# Ensure backend package is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
import orjson

from backend.config import settings
from backend.schemas import QueryResponse, HealthResponse, ErrorResponse
from backend.orchestrator import orchestrate_query
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
    except Exception as exc:
        logger.warning("Health check degraded: %s", exc)
        index_ok = False
        embedder_ok = False

    return HealthResponse(
        status="ready" if (index_ok and embedder_ok) else "degraded",
        index_version=settings.index_version,
        embedding_model=settings.embedding_model,
        index_ready=index_ok,
        embedder_ready=embedder_ok,
    )


@app.post("/api/query", response_model=QueryResponse)
async def query(
    audio: UploadFile = File(...),
    language: str = Form(default="auto"),
) -> QueryResponse:
    """
    Main RAG pipeline endpoint.
    Accepts audio upload, returns grounded answer + sources + latency.
    """
    # --- Input validation ---
    if audio.size and audio.size > settings.max_audio_bytes:
        raise HTTPException(status_code=413, detail="Audio file too large")

    content_type = (audio.content_type or "").lower()
    if not any(ct in content_type for ct in ("audio", "octet-stream", "webm", "wav", "ogg", "mp4", "m4a", "aac", "opus")):
        logger.warning("Unexpected content_type: %s", content_type)


    audio_bytes = await audio.read()
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty audio payload")

    result = await orchestrate_query(audio_bytes=audio_bytes, language=language)
    return result


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

