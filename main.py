"""
MangoVoice — Root module entry point.
Supports direct execution and $env:EVAL_EMBEDDER_MODULE = "main" / $env:EVAL_GENERATOR_MODULE = "main".
"""
from app.embedder import embed, embed_one, get_model, LATENCY_BUDGET_MS
from app.generator import generate_answer, Answer

__all__ = [
    "embed",
    "embed_one",
    "get_model",
    "generate_answer",
    "Answer",
    "LATENCY_BUDGET_MS",
]
