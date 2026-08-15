"use client";

import { useState } from "react";
import { ChevronDown, ChevronUp, ExternalLink } from "lucide-react";

interface Source {
  chunk_id: string;
  parent_id: string;
  score: number;
  rrf_score?: number;
  text: string;
  language: string;
  strategy: string;
}

interface SourceInspectorProps {
  sources: Source[];
  visible: boolean;
}

const STRATEGY_LABELS: Record<string, string> = {
  canonical: "Canonical",
  sentence_window: "Sentence Window",
  fixed_token: "Fixed Token",
  semantic: "Semantic",
  parent_child: "Parent-Child",
};

const LANG_BADGES: Record<string, string> = {
  hi: "हि",
  "hi-IN": "हि",
  en: "EN",
  "en-IN": "EN",
};

export default function SourceInspector({ sources, visible }: SourceInspectorProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);

  if (!visible || !sources.length) return null;

  return (
    <div className="card-glass rounded-card overflow-hidden animate-fade-in" style={{ animationDelay: "200ms" }}>
      {/* Toggle header */}
      <button
        id="sources-toggle"
        onClick={() => setIsOpen((o) => !o)}
        className="w-full flex items-center justify-between p-5 text-left hover:bg-cream/5 transition-colors"
      >
        <div className="flex items-center gap-3">
          <div className="eyebrow text-cream/60">Sources</div>
          <span className="bg-sunshine text-forest text-xs font-bold rounded-pill px-2 py-0.5 font-mono">
            {sources.length}
          </span>
        </div>
        {isOpen ? (
          <ChevronUp size={16} className="text-cream/50" />
        ) : (
          <ChevronDown size={16} className="text-cream/50" />
        )}
      </button>

      {/* Sources list */}
      {isOpen && (
        <div className="border-t border-cream/10">
          {sources.map((src, idx) => (
            <div key={src.chunk_id} className="border-b border-cream/10 last:border-0">
              {/* Source header */}
              <button
                onClick={() => setExpandedIdx(expandedIdx === idx ? null : idx)}
                className="w-full flex items-start gap-4 p-4 text-left hover:bg-cream/5 transition-colors"
              >
                {/* Rank badge */}
                <div className="w-6 h-6 rounded-full bg-sunshine/10 border border-sunshine/20 flex items-center justify-center flex-shrink-0 mt-0.5">
                  <span className="text-sunshine text-xs font-bold font-mono">
                    {idx + 1}
                  </span>
                </div>

                <div className="flex-1 min-w-0">
                  {/* Metadata row */}
                  <div className="flex items-center gap-2 flex-wrap mb-1.5">
                    {/* Language */}
                    <span className="text-xs font-mono text-forest bg-sunshine/80 rounded px-1.5 py-0.5 font-bold">
                      {LANG_BADGES[src.language] || src.language?.toUpperCase() || "EN"}
                    </span>
                    {/* Strategy */}
                    <span className="text-xs font-mono text-cream/50 bg-cream/5 rounded px-1.5 py-0.5">
                      {STRATEGY_LABELS[src.strategy] || src.strategy}
                    </span>
                    {/* Score */}
                    <span className="text-xs font-mono text-muted-text ml-auto">
                      score: {(src.rrf_score ?? src.score).toFixed(4)}
                    </span>
                  </div>

                  {/* Passage preview */}
                  <p className="text-cream/70 text-xs font-mono leading-relaxed line-clamp-2">
                    {src.text}
                  </p>

                  {/* Chunk ID */}
                  <div className="mt-1 text-muted-text text-xs font-mono truncate">
                    {src.chunk_id}
                  </div>
                </div>

                {expandedIdx === idx ? (
                  <ChevronUp size={14} className="text-cream/30 flex-shrink-0 mt-1" />
                ) : (
                  <ChevronDown size={14} className="text-cream/30 flex-shrink-0 mt-1" />
                )}
              </button>

              {/* Expanded view */}
              {expandedIdx === idx && (
                <div className="px-4 pb-4 ml-10 border-t border-cream/5 pt-3">
                  <p className="text-cream/80 text-sm font-mono leading-relaxed">
                    {src.text}
                  </p>
                  <div className="mt-3 grid grid-cols-2 gap-2 text-xs font-mono text-muted-text">
                    <div>
                      <span className="text-cream/30">Chunk ID: </span>
                      <span className="text-cream/60 break-all">{src.chunk_id}</span>
                    </div>
                    <div>
                      <span className="text-cream/30">Parent: </span>
                      <span className="text-cream/60 break-all">{src.parent_id}</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
