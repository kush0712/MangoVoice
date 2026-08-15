"use client";

import { ChevronDown, ChevronUp } from "lucide-react";
import { useState } from "react";

interface LatencyEntry {
  label: string;
  value: number;
  isTotal?: boolean;
  highlight?: boolean;
}

interface LatencyPanelProps {
  stt_ms?: number;
  normalization_ms?: number;
  embedding_ms?: number;
  retrieval_ms?: number;
  safety_ms?: number;
  generation_ms?: number;
  grounding_ms?: number;
  rag_core_ms?: number;
  full_e2e_ms?: number;
  visible: boolean;
}

function LatencyBar({ value, max }: { value: number; max: number }) {
  const pct = Math.min((value / max) * 100, 100);
  return (
    <div className="h-1 bg-cream/5 rounded-full flex-1 max-w-20">
      <div
        className="h-1 rounded-full bg-sunshine/60 transition-all duration-700"
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}

export default function LatencyPanel({
  stt_ms = 0,
  normalization_ms = 0,
  embedding_ms = 0,
  retrieval_ms = 0,
  safety_ms = 0,
  generation_ms = 0,
  grounding_ms = 0,
  rag_core_ms = 0,
  full_e2e_ms = 0,
  visible,
}: LatencyPanelProps) {
  const [showDetail, setShowDetail] = useState(false);

  if (!visible) return null;

  const entries: LatencyEntry[] = [
    { label: "STT (Sarvam)", value: stt_ms },
    { label: "Normalization", value: normalization_ms },
    { label: "Embedding", value: embedding_ms },
    { label: "Retrieval", value: retrieval_ms },
    { label: "Safety", value: safety_ms },
    { label: "Generation", value: generation_ms },
    { label: "Grounding", value: grounding_ms },
  ];

  const maxVal = Math.max(...entries.map((e) => e.value), 1);

  const isUnder200 = rag_core_ms > 0 && rag_core_ms < 200;

  return (
    <div className="card-glass rounded-card overflow-hidden animate-fade-in" style={{ animationDelay: "300ms" }}>
      {/* Summary row */}
      <div className="p-5">
        <div className="flex items-center justify-between mb-4">
          <div className="eyebrow text-cream/60">Latency</div>
          {rag_core_ms > 0 && (
            <div
              className={`text-xs font-mono font-bold px-2 py-1 rounded-tag ${
                isUnder200
                  ? "bg-sunshine/15 text-sunshine border border-sunshine/20"
                  : "bg-hibiscus/15 text-hibiscus border border-hibiscus/20"
              }`}
            >
              {isUnder200 ? "⚡ Under 200ms" : `${rag_core_ms.toFixed(0)}ms`}
            </div>
          )}
        </div>

        {/* Main metrics */}
        <div className="grid grid-cols-2 gap-3">
          {[
            { label: "RAG Core", value: rag_core_ms, highlight: true },
            { label: "Full E2E", value: full_e2e_ms },
          ].map(({ label, value, highlight }) => (
            <div
              key={label}
              className={`rounded-tag p-3 ${
                highlight
                  ? "bg-sunshine/10 border border-sunshine/20"
                  : "bg-cream/5 border border-cream/10"
              }`}
            >
              <div className="text-muted-text text-xs font-mono uppercase tracking-wider mb-1">
                {label}
              </div>
              <div className={`text-xl font-bold font-mono ${highlight ? "text-sunshine" : "text-cream"}`}>
                {value > 0 ? `${value.toFixed(0)}` : "—"}
                {value > 0 && <span className="text-sm font-normal ml-0.5 opacity-70">ms</span>}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Detail toggle */}
      <button
        id="latency-detail-toggle"
        onClick={() => setShowDetail((s) => !s)}
        className="w-full flex items-center justify-between px-5 py-3 text-xs font-mono text-muted-text hover:text-cream/60 hover:bg-cream/5 transition-colors border-t border-cream/10"
      >
        <span className="uppercase tracking-wider">Stage breakdown</span>
        {showDetail ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
      </button>

      {showDetail && (
        <div className="px-5 pb-5 space-y-2.5">
          {entries.map((entry) => (
            <div key={entry.label} className="flex items-center gap-3">
              <div className="text-xs font-mono text-muted-text w-28 flex-shrink-0">
                {entry.label}
              </div>
              <LatencyBar value={entry.value} max={maxVal} />
              <div className="text-xs font-mono text-cream/60 w-14 text-right flex-shrink-0">
                {entry.value > 0 ? `${entry.value.toFixed(1)}ms` : "—"}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
