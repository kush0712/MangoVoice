"""
MangoVoice — All Pydantic v2 schemas for the RAG pipeline.
Every stage input/output is typed. No raw dicts cross stage boundaries.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────────────────

class PipelineStatus(str, Enum):
    IDLE = "idle"
    RECEIVED = "received"
    TRANSCRIBING = "transcribing"
    TRANSCRIBED = "transcribed"
    SAFETY_RETRIEVAL = "safety_retrieval"
    CONFIDENCE_GATE = "confidence_gate"
    GENERATING = "generating"
    GROUNDING = "grounding"
    ANSWERED = "answered"
    REFUSED = "refused"
    ERROR = "error"


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    REFUSED = "refused"


class RefusalReason(str, Enum):
    LOW_CONFIDENCE = "low_confidence"
    SAFETY_VIOLATION = "safety_violation"
    UNSAFE_INPUT = "unsafe_input"
    PROMPT_INJECTION = "prompt_injection"
    NO_EVIDENCE = "no_evidence"
    GROUNDING_FAILED = "grounding_failed"
    STT_FAILED = "stt_failed"
    GENERATION_UNAVAILABLE = "generation_unavailable"
    TIMEOUT = "timeout"


class ChunkingStrategy(str, Enum):
    CANONICAL = "canonical"
    SENTENCE_WINDOW = "sentence_window"
    FIXED_TOKEN = "fixed_token"
    SEMANTIC = "semantic"
    PARENT_CHILD = "parent_child"


# ── Stage models ─────────────────────────────────────────────────────────────

class TranscriptResult(BaseModel):
    text: str
    language: str | None = None
    confidence: float | None = None
    duration_ms: float | None = None


class RetrievalSource(BaseModel):
    chunk_id: str
    parent_id: str
    score: float
    raw_dense_score: float = 0.0  # preserved cosine similarity (1 - cosine_distance), never overwritten by RRF
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rrf_score: float | None = None
    text: str
    language: str = "en"
    strategy: ChunkingStrategy = ChunkingStrategy.PARENT_CHILD
    query_id: str | None = None


class RetrievalResult(BaseModel):
    sources: list[RetrievalSource] = Field(default_factory=list)
    top_score: float = 0.0
    margin: float = 0.0           # top1 - top2 score
    confidence: float = 0.0       # composite confidence 0-1
    dense_bm25_agree: bool = False
    supporting_count: int = 0     # candidates with score > threshold


class GuardrailResult(BaseModel):
    passed: bool
    refusal_reason: RefusalReason | None = None
    message: str | None = None
    injection_detected: bool = False
    unsafe_detected: bool = False
    latency_ms: float = 0.0


class GenerationResult(BaseModel):
    status: Literal["answered", "refused"]
    answer: str | None = None
    cited_chunk_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    refusal_reason: RefusalReason | None = None
    regenerated: bool = False


class GroundingResult(BaseModel):
    passed: bool
    score: float = 0.0
    sentence_scores: list[float] = Field(default_factory=list)
    entity_overlap: float = 0.0
    citation_valid: bool = True


class LatencyMetrics(BaseModel):
    stt_ms: float = 0.0
    normalization_ms: float = 0.0
    embedding_ms: float = 0.0
    retrieval_ms: float = 0.0
    safety_ms: float = 0.0
    generation_ms: float = 0.0
    grounding_ms: float = 0.0
    rag_core_ms: float = 0.0
    full_e2e_ms: float = 0.0


# ── API response models ───────────────────────────────────────────────────────

class QueryResponse(BaseModel):
    request_id: str
    status: PipelineStatus
    transcript: str | None = None
    language: str | None = None
    answer: str | None = None
    confidence: ConfidenceLevel = ConfidenceLevel.REFUSED
    confidence_score: float = 0.0
    sources: list[RetrievalSource] = Field(default_factory=list)
    refusal_reason: RefusalReason | None = None
    refusal_message: str | None = None
    latency: LatencyMetrics = Field(default_factory=LatencyMetrics)
    grounding_score: float | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    status: Literal["ready", "degraded"]
    index_version: str
    embedding_model: str
    index_ready: bool
    embedder_ready: bool
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    request_id: str | None = None


# ── Benchmark / evaluation models ─────────────────────────────────────────────

class BenchmarkRow(BaseModel):
    query_id: str
    language: str
    retrieval_strategy: ChunkingStrategy
    embedding_ms: float
    retrieval_ms: float
    safety_ms: float
    generation_ms: float
    grounding_ms: float
    rag_core_ms: float
    full_e2e_ms: float
    status: PipelineStatus
    refusal_reason: RefusalReason | None = None
    recall_at_5: float | None = None
    recall_at_10: float | None = None
    mrr_at_10: float | None = None
