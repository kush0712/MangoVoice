"""
MangoVoice — Application & Evaluation Configuration.
Declares backend type and settings expected by rag-local-eval-loop.
"""
from __future__ import annotations

import os

# Evaluation harness backend identifiers
GENERATION_BACKEND = "groq"
GENERATION_BACKEND_NAME = "Groq Hybrid (allam-2-7b / Extractive Fallback)"
GENERATION_MODEL = os.getenv("GROQ_MODEL", "groq/compound-mini")
LOCAL_GENERATION_MODEL = "extractive-fast-qa"

# Evaluation budgets
LATENCY_BUDGET_MS = 50
EMBEDDING_LATENCY_BUDGET_MS = 50
GENERATION_LATENCY_BUDGET_MS = 1500

# Model identifiers
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
GROQ_MODEL = os.getenv("GROQ_MODEL", "groq/compound-mini")
