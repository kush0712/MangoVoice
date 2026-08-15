"use client";

import { MessageSquare, Clock, Trash2, ChevronDown, ChevronUp } from "lucide-react";
import { useState } from "react";

export interface HistoryEntry {
  id: string;
  transcript: string;
  answer: string | null;
  confidence: "high" | "medium" | "refused";
  timestamp: number;
  rag_core_ms?: number;
}

interface HistoryPanelProps {
  entries: HistoryEntry[];
  onSelect: (entry: HistoryEntry) => void;
  onClear: () => void;
}

function formatTime(ts: number): string {
  const d = new Date(ts);
  return d.toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit" });
}

const CONF_DOT: Record<string, string> = {
  high: "bg-sunshine",
  medium: "bg-sunshine/50",
  refused: "bg-hibiscus",
};

export default function HistoryPanel({ entries, onSelect, onClear }: HistoryPanelProps) {
  const [isOpen, setIsOpen] = useState(false);

  if (entries.length === 0) return null;

  return (
    <div className="card-glass rounded-card overflow-hidden">
      <button
        id="history-toggle"
        onClick={() => setIsOpen((o) => !o)}
        className="w-full flex items-center justify-between p-4 text-left hover:bg-cream/5 transition-colors"
      >
        <div className="flex items-center gap-2.5">
          <MessageSquare size={14} className="text-cream/40" />
          <span className="text-xs font-mono text-cream/50 uppercase tracking-wider">
            Recent ({entries.length})
          </span>
        </div>
        {isOpen ? (
          <ChevronUp size={12} className="text-cream/30" />
        ) : (
          <ChevronDown size={12} className="text-cream/30" />
        )}
      </button>

      {isOpen && (
        <div className="border-t border-cream/10">
          {/* Clear button */}
          <div className="flex justify-end px-4 py-2 border-b border-cream/10">
            <button
              onClick={onClear}
              className="flex items-center gap-1.5 text-xs font-mono text-muted-text hover:text-hibiscus transition-colors"
            >
              <Trash2 size={11} />
              Clear history
            </button>
          </div>

          {/* Entries */}
          <div className="max-h-64 overflow-y-auto">
            {entries.map((entry) => (
              <button
                key={entry.id}
                onClick={() => onSelect(entry)}
                className="w-full flex items-start gap-3 p-4 text-left hover:bg-cream/5 transition-colors border-b border-cream/5 last:border-0"
              >
                <span
                  className={`w-2 h-2 rounded-full mt-1 flex-shrink-0 ${CONF_DOT[entry.confidence]}`}
                />
                <div className="flex-1 min-w-0">
                  <p className="text-cream/70 text-xs font-mono truncate">
                    {entry.transcript}
                  </p>
                  {entry.answer && (
                    <p className="text-muted-text text-xs font-mono truncate mt-0.5">
                      {entry.answer}
                    </p>
                  )}
                </div>
                <div className="flex items-center gap-1 text-muted-text text-xs font-mono flex-shrink-0">
                  <Clock size={10} />
                  {formatTime(entry.timestamp)}
                </div>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
