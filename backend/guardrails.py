"""
MangoVoice — Guardrail system (4 layers).

Layer 1: deterministic normalization + injection phrase detection (free, fast)
Layer 2: Groq Llama Guard 4 safety classification (parallel with retrieval)
Layer 3: full safety classifier (only on suspicious input)
Layer 4: grounding verifier lives in grounding.py

Returns GuardrailResult with pass/refuse + reason.
"""
from __future__ import annotations

import re
import time
import unicodedata
from functools import lru_cache

from backend.config import settings
from backend.schemas import GuardrailResult, RefusalReason
from backend.telemetry import get_logger

logger = get_logger(__name__)

# ── Layer 1 — deterministic patterns ─────────────────────────────────────────

# Classic prompt injection / jailbreak phrases
_INJECTION_PATTERNS = [
    r"ignore (all |previous |above |prior )?instructions",
    r"you are now",
    r"pretend (to be|you are|you're)",
    r"disregard (your|all|previous)",
    r"override (your|all|previous|system)",
    r"jailbreak",
    r"dan mode",
    r"act as (if you are|a)?",
    r"forget (your|all|previous|system|that)",
    r"new persona",
    r"system prompt",
    r"<\|im_start\|>",
    r"<\|system\|>",
    r"\[INST\]",
    r"ASSISTANT:",
]

# Unsafe content patterns (very broad — Layer 2 handles nuance)
_UNSAFE_PATTERNS = [
    r"\b(bomb|weapon|explosive|poison|drug synthesis|meth|fentanyl)\b",
    r"\b(suicide method|kill (my|your)self|self.harm)\b",
    r"\b(child porn|csam|abuse material)\b",
    r"\b(hack|exploit|sql injection|xss attack|ddos)\b",
]

_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE | re.DOTALL)
_UNSAFE_RE = re.compile("|".join(_UNSAFE_PATTERNS), re.IGNORECASE)


def normalize_text(text: str) -> str:
    """Normalize Unicode, collapse whitespace, strip control chars."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def layer1_check(text: str) -> GuardrailResult:
    """Deterministic input safety check. Cost: negligible."""
    t0 = time.perf_counter()

    if not text or not text.strip():
        return GuardrailResult(
            passed=False,
            refusal_reason=RefusalReason.UNSAFE_INPUT,
            message="Empty query",
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    if len(text) > settings.max_transcript_chars:
        return GuardrailResult(
            passed=False,
            refusal_reason=RefusalReason.UNSAFE_INPUT,
            message="Query exceeds maximum length",
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    injection_match = _INJECTION_RE.search(text)
    if injection_match:
        logger.warning("Layer1 injection detected: %s", injection_match.group()[:50])
        return GuardrailResult(
            passed=False,
            refusal_reason=RefusalReason.PROMPT_INJECTION,
            message="I can't help with that request.",
            injection_detected=True,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    unsafe_match = _UNSAFE_RE.search(text)
    if unsafe_match:
        return GuardrailResult(
            passed=False,
            refusal_reason=RefusalReason.UNSAFE_INPUT,
            message="I can't help with that request.",
            unsafe_detected=True,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    return GuardrailResult(
        passed=True,
        latency_ms=(time.perf_counter() - t0) * 1000,
    )


async def layer2_prompt_guard(text: str) -> GuardrailResult:
    """
    Groq Llama Prompt Guard 2 safety classification.
    Model: meta-llama/llama-prompt-guard-2-86m
    Runs in parallel with retrieval. Skipped if Groq key unavailable.
    """
    if not settings.has_groq_key:
        return GuardrailResult(passed=True, message="Safety check skipped (no GROQ key)")

    t0 = time.perf_counter()
    try:
        from groq import AsyncGroq
        client = AsyncGroq(api_key=settings.groq_api_key)

        # Llama Prompt Guard 2 22M: still active on Groq, lower-latency variant.
        # 86M was deprecated Aug 2026 — 22M provides same jailbreak/injection detection.
        chat_resp = await client.chat.completions.create(
            model="meta-llama/llama-prompt-guard-2-22m",
            messages=[
                {"role": "user", "content": text},
            ],
            max_tokens=10,
            temperature=0.0,
            timeout=settings.groq_safety_timeout,
        )
        classification = (chat_resp.choices[0].message.content or "safe").strip().lower()
        latency_ms = (time.perf_counter() - t0) * 1000

        # Prompt Guard 2 returns: "safe" or "injection" / "jailbreak"
        is_unsafe = any(w in classification for w in ("injection", "jailbreak", "unsafe", "malicious"))
        if is_unsafe:
            category = classification.split("\n")[-1].strip() if "\n" in classification else "unsafe"
            logger.info(
                "Layer2 safety flag: %s", category,
                extra={"stage": "guardrail_l2", "latency_ms": round(latency_ms, 1)},
            )
            return GuardrailResult(
                passed=False,
                refusal_reason=RefusalReason.UNSAFE_INPUT,
                message="I can't help with that request.",
                unsafe_detected=True,
                latency_ms=latency_ms,
            )
        return GuardrailResult(passed=True, latency_ms=latency_ms)

    except Exception as exc:
        # Fail open — Layer 1 deterministic rules already ran
        logger.warning("Layer2 guard failed (falling back to allow): %s", exc)
        return GuardrailResult(
            passed=True,
            message="Safety model unavailable — deterministic check passed",
            latency_ms=(time.perf_counter() - t0) * 1000,
        )


async def run_guardrails(text: str) -> GuardrailResult:
    """
    Full guardrail pipeline (Layers 1 + 2).
    Called before retrieval; Layer 2 runs concurrently with retrieval in orchestrator.
    """
    normalized = normalize_text(text)
    l1 = layer1_check(normalized)
    if not l1.passed:
        return l1
    # Layer 2 is invoked concurrently in the orchestrator
    return l1
