/**
 * MangoVoice — API client.
 * Typed wrappers for /api/query, /api/query/text, /api/health.
 */

export interface Source {
  chunk_id: string;
  parent_id: string;
  score: number;
  rrf_score?: number;
  text: string;
  language: string;
  strategy: string;
}

export interface LatencyMetrics {
  stt_ms: number;
  normalization_ms: number;
  embedding_ms: number;
  retrieval_ms: number;
  safety_ms: number;
  generation_ms: number;
  grounding_ms: number;
  rag_core_ms: number;
  full_e2e_ms: number;
}

export interface QueryResponse {
  request_id: string;
  status: string;
  transcript: string | null;
  language: string | null;
  answer: string | null;
  confidence: "high" | "medium" | "refused";
  confidence_score: number;
  sources: Source[];
  refusal_reason: string | null;
  refusal_message: string | null;
  latency: LatencyMetrics;
  grounding_score: number | null;
}

export interface HealthResponse {
  status: "ready" | "degraded";
  index_version: string;
  embedding_model: string;
  index_ready: boolean;
  embedder_ready: boolean;
}

const BASE = process.env.NEXT_PUBLIC_API_BASE || "";

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const json = await res.json();
      detail = json.detail || json.message || detail;
    } catch {}
    throw new Error(`API error ${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

export async function queryAudio(
  audioBlob: Blob,
  language: string = "auto"
): Promise<QueryResponse> {
  const formData = new FormData();
  formData.append("audio", audioBlob, "recording.webm");
  formData.append("language", language);

  const res = await fetch(`${BASE}/api/query`, {
    method: "POST",
    body: formData,
  });
  return handleResponse<QueryResponse>(res);
}

export async function queryText(
  text: string,
  language: string = "auto"
): Promise<QueryResponse> {
  const res = await fetch(`${BASE}/api/query/text`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, language }),
  });
  return handleResponse<QueryResponse>(res);
}

export async function checkHealth(): Promise<HealthResponse> {
  const res = await fetch(`${BASE}/api/health`, { cache: "no-store" });
  return handleResponse<HealthResponse>(res);
}
