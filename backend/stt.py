"""
MangoVoice — Sarvam Saaras v3 REST STT adapter.

Sends recorded audio bytes to Sarvam's synchronous REST API.
Returns a TranscriptResult. Never exposes raw exceptions to callers.
1 retry on transient network failure. Hard timeout of sarvam_timeout seconds.
"""
from __future__ import annotations

import time
from io import BytesIO

import httpx
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

from backend.config import settings
from backend.schemas import TranscriptResult, RefusalReason
from backend.telemetry import get_logger

logger = get_logger(__name__)

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"


class STTError(Exception):
    def __init__(self, reason: RefusalReason, message: str):
        self.reason = reason
        self.message = message
        super().__init__(message)


def _detect_audio_format(audio_bytes: bytes) -> tuple[str, str]:
    """Detect file extension and mime-type from header bytes."""
    if audio_bytes.startswith(b"RIFF") and b"WAVE" in audio_bytes[:16]:
        return "audio.wav", "audio/wav"
    if audio_bytes.startswith(b"\x1a\x45\xdf\xa3"):
        return "audio.webm", "audio/webm"
    if audio_bytes.startswith(b"OggS"):
        return "audio.ogg", "audio/ogg"
    if len(audio_bytes) >= 12 and (
        audio_bytes[4:8] in (b"ftyp", b"moov", b"mdat") or
        audio_bytes[8:12] in (b"ftyp", b"mp42", b"isom", b"M4A ", b"mp41")
    ):
        return "audio.mp4", "audio/mp4"
    if audio_bytes.startswith(b"\xff\xf1") or audio_bytes.startswith(b"\xff\xf9"):
        return "audio.aac", "audio/aac"
    return "audio.webm", "audio/webm"


_http_client: httpx.AsyncClient | None = None

def get_stt_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            timeout=settings.sarvam_timeout,
            limits=httpx.Limits(max_keepalive_connections=10, keepalive_expiry=60.0),
        )
    return _http_client

@retry(
    stop=stop_after_attempt(2),
    wait=wait_fixed(0.5),
    retry=retry_if_exception_type((httpx.NetworkError, httpx.TimeoutException)),
    reraise=False,
)
async def _call_sarvam(audio_bytes: bytes, language: str) -> dict:
    """Raw Sarvam REST call — retried on transient errors."""
    if not settings.has_sarvam_key:
        raise STTError(RefusalReason.STT_FAILED, "SARVAM_API_KEY not configured")

    filename, mime = _detect_audio_format(audio_bytes)
    files = {"file": (filename, BytesIO(audio_bytes), mime)}
    data = {
        "model": settings.sarvam_model,
        "with_timestamps": "false",
        "with_disfluencies": "false",
    }

    # Use codemix / auto-detection unless a specific language is explicitly set
    if language and language not in ("auto", "unknown"):
        data["language_code"] = language
    else:
        # Sarvam Saaras v3: 'unknown' enables automatic language identification
        # (transcribes English as English text, Hindi as Hindi text, Hinglish as codemixed)
        data["language_code"] = "unknown"

    client = get_stt_client()
    resp = await client.post(
        SARVAM_STT_URL,
        files=files,
        data=data,
        headers={"api-subscription-key": settings.sarvam_api_key},
    )
    resp.raise_for_status()
    return resp.json()



async def transcribe(audio_bytes: bytes, language: str = "auto") -> TranscriptResult:
    """
    Transcribe audio bytes using Sarvam Saaras v3.
    Returns TranscriptResult. Raises STTError on failure.
    """
    if not audio_bytes:
        raise STTError(RefusalReason.STT_FAILED, "Empty audio payload")

    t0 = time.perf_counter()
    try:
        result = await _call_sarvam(audio_bytes, language)
    except STTError:
        raise
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if status in (401, 403):
            raise STTError(RefusalReason.STT_FAILED, "Invalid Sarvam API key")
        if status == 422:
            raise STTError(RefusalReason.STT_FAILED, "Malformed audio or unsupported format")
        raise STTError(RefusalReason.STT_FAILED, f"Sarvam API error {status}")
    except (httpx.TimeoutException, httpx.NetworkError) as exc:
        raise STTError(RefusalReason.STT_FAILED, "Sarvam STT timed out. Please try again.")
    except Exception as exc:
        logger.warning("Unexpected STT error: %s", exc)
        raise STTError(RefusalReason.STT_FAILED, "Speech-to-text service unavailable")

    duration_ms = (time.perf_counter() - t0) * 1000

    transcript = (result.get("transcript") or "").strip()
    if not transcript:
        raise STTError(RefusalReason.STT_FAILED, "Could not transcribe the recording. Please try again.")

    detected_lang = result.get("language_code") or language or None

    logger.info(
        "STT complete",
        extra={"stage": "stt", "latency_ms": round(duration_ms, 1)},
    )
    return TranscriptResult(
        text=transcript,
        language=detected_lang,
        confidence=result.get("confidence"),
        duration_ms=duration_ms,
    )


async def demo_transcribe(text: str) -> TranscriptResult:
    """
    Pass-through for text-input demo mode (no audio needed).
    """
    return TranscriptResult(text=text, language="en", confidence=1.0, duration_ms=0.0)
