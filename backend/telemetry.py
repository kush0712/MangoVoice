"""
MangoVoice — Structured JSON logger.
Records request ID, stage timings, status, model IDs.
Never logs raw audio or API keys.
"""
from __future__ import annotations

import logging
import sys
import time
import uuid
from contextvars import ContextVar
from typing import Any

import orjson

# Per-request context variable for request ID
_request_id_var: ContextVar[str] = ContextVar("request_id", default="")


def new_request_id() -> str:
    rid = str(uuid.uuid4())
    _request_id_var.set(rid)
    return rid


def current_request_id() -> str:
    return _request_id_var.get() or str(uuid.uuid4())


class ORJSONFormatter(logging.Formatter):
    """Emit log records as compact JSON lines."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.time(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": current_request_id(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # Attach any extra fields passed via record.__dict__
        for key in ("stage", "latency_ms", "status", "model", "n_sources", "error_code"):
            val = record.__dict__.get(key)
            if val is not None:
                payload[key] = val
        return orjson.dumps(payload).decode()


def get_logger(name: str) -> logging.Logger:
    from backend.config import settings

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(ORJSONFormatter())
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    return logger
