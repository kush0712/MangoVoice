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

const RAW_BASE = process.env.NEXT_PUBLIC_API_BASE || "";
const BASE = RAW_BASE.endsWith("/") ? RAW_BASE.slice(0, -1) : RAW_BASE;

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
  let filename = "recording.webm";
  if (audioBlob.type.includes("mp4") || audioBlob.type.includes("m4a") || audioBlob.type.includes("aac")) {
    filename = "recording.mp4";
  } else if (audioBlob.type.includes("wav")) {
    filename = "recording.wav";
  } else if (audioBlob.type.includes("ogg")) {
    filename = "recording.ogg";
  }
  formData.append("audio", audioBlob, filename);
  formData.append("language", language);

  try {
    const t0 = performance.now();
    let sttData: { transcript: string; language: string };
    
    const directKey = process.env.NEXT_PUBLIC_SARVAM_API_KEY;
    if (directKey) {
      // 🚀 NUCLEAR OPTION: Bypassing Vercel Serverless completely.
      // Browser talks directly to Sarvam for 0ms cold-start and absolute minimum latency.
      const sarvamFormData = new FormData();
      sarvamFormData.append("file", audioBlob, filename);
      sarvamFormData.append("model", "saaras:v3");
      sarvamFormData.append("with_timestamps", "false");
      sarvamFormData.append("with_disfluencies", "false");
      sarvamFormData.append("language_code", language === "auto" ? "unknown" : language);

      const sarvamRes = await fetch("https://api.sarvam.ai/speech-to-text", {
        method: "POST",
        headers: { "api-subscription-key": directKey },
        body: sarvamFormData,
      });

      if (!sarvamRes.ok) {
        const errText = await sarvamRes.text();
        throw new Error(`Direct Sarvam Error: ${errText}`);
      }
      const rawJson = await sarvamRes.json();
      sttData = { transcript: rawJson.transcript, language: rawJson.language_code };
    } else {
      // Fallback to Vercel Serverless if they haven't set the NEXT_PUBLIC key yet
      const sttRes = await fetch(`/api/stt`, {
        method: "POST",
        body: formData,
      });
      if (!sttRes.ok) {
        const errText = await sttRes.text();
        throw new Error(`STT Edge Error: ${errText}`);
      }
      sttData = await sttRes.json();
    }
    const stt_ms = performance.now() - t0;
    
    if (!sttData.transcript) {
      return {
        request_id: "error",
        status: "error",
        refusal_reason: "stt_failed",
        refusal_message: "Could not transcribe the recording. Please try again.",
        transcript: null,
        language: null,
        answer: null,
        grounding_score: null,
        confidence: "refused",
        confidence_score: 0,
        sources: [],
        latency: { 
          stt_ms, 
          normalization_ms: 0,
          embedding_ms: 0,
          retrieval_ms: 0, 
          safety_ms: 0,
          generation_ms: 0, 
          grounding_ms: 0, 
          rag_core_ms: 0,
          full_e2e_ms: stt_ms 
        }
      };
    }

    // Pass the transcript to the RAG backend via the proxy
    const ragRes = await queryText(sttData.transcript, sttData.language || language);
    
    // Inject the STT latency and update the full E2E latency
    if (ragRes.latency) {
      ragRes.latency.stt_ms = stt_ms;
      ragRes.latency.full_e2e_ms = stt_ms + ragRes.latency.full_e2e_ms;
    }
    
    // Pass the actual language detected by the Edge function
    ragRes.language = sttData.language || ragRes.language;
    return ragRes;

  } catch (err: unknown) {
    if (err instanceof Error) {
      if (err.message.includes("Load failed") || err.message.includes("Failed to fetch") || err.message.includes("NetworkError")) {
        throw new Error(`Unable to connect to STT Edge Server. Please ensure you are connected to the internet.`);
      }
      throw err;
    }
    throw new Error("Network request failed");
  }
}

export async function queryText(
  text: string,
  language: string = "auto"
): Promise<QueryResponse> {
  try {
    const res = await fetch(`/api/query/text`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, language }),
    });
    return await handleResponse<QueryResponse>(res);
  } catch (err: unknown) {
    if (err instanceof Error) {
      if (err.message.includes("Load failed") || err.message.includes("Failed to fetch") || err.message.includes("NetworkError")) {
        throw new Error(`Unable to connect to backend server (proxied via /api/query/text). If in development, verify it's running on http://127.0.0.1:8000. In production, check NEXT_PUBLIC_API_BASE in Vercel.`);
      }
      throw err;
    }
    throw new Error("Network request failed");
  }
}

export async function checkHealth(): Promise<HealthResponse> {
  const res = await fetch(`/api/health`, { cache: "no-store" });
  return handleResponse<HealthResponse>(res);
}

