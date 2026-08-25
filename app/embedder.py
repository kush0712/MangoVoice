"""
MangoVoice — Eval Loop Embedder Module.
Conforms to TARGET_INTERFACE.md for rag-local-eval-loop.
"""
from __future__ import annotations

import os
import sys
import numpy as np

# Ensure repository root is on sys.path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from backend.retrieval.embeddings import get_embedder

LATENCY_BUDGET_MS = 200


def get_model():
    """Warms up and initializes the embedding model singleton."""
    return get_embedder()


def embed(texts: list[str]) -> np.ndarray:
    """Embed a list of texts into a 2D float32 numpy array of shape (len(texts), dim)."""
    embedder = get_embedder()
    return embedder.embed(texts)


def embed_one(text: str) -> np.ndarray:
    """Embed a single string into a 1D float32 numpy array of shape (dim,)."""
    embedder = get_embedder()
    return embedder.embed_one(text)
