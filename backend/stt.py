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

    files = {"file": ("audio.webm", BytesIO(audio_bytes), "audio/webm")}
    data = {
        "model": settings.sarvam_model,
        "with_timestamps": "false",
        "with_disfluencies": "false",
    }

    # Use codemix mode unless a specific language is explicitly set
    if language and language != "auto":
        data["language_code"] = language
    else:
        # Sarvam codemix: handles Hindi/English/Hinglish automatically
        data["language_code"] = "hi-IN"  # default hint; sarvam handles mixing

    async with httpx.AsyncClient(timeout=settings.sarvam_timeout) as client:
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
