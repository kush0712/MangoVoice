"use client";

import { Check, Loader2, AlertCircle } from "lucide-react";

export type StepStatus = "pending" | "active" | "done" | "error" | "skipped";

export interface PipelineStep {
  id: string;
  label: string;
  emoji: string;
  status: StepStatus;
}

interface PipelineStatusProps {
  steps: PipelineStep[];
  visible: boolean;
}

const STEP_ICON = {
  pending: null,
  active: <Loader2 size={14} className="animate-spin text-sunshine" />,
  done: <Check size={14} className="text-sunshine" />,
  error: <AlertCircle size={14} className="text-hibiscus" />,
  skipped: null,
};

export default function PipelineStatus({ steps, visible }: PipelineStatusProps) {
  if (!visible) return null;

  return (
    <div className="card-glass rounded-card p-5 animate-fade-in">
      <div className="eyebrow mb-4 text-cream opacity-70">Pipeline</div>
      <div className="flex flex-col gap-2.5">
        {steps.map((step, idx) => (
          <div
            key={step.id}
            className={`pipeline-step ${step.status}`}
            style={{
              animationDelay: `${idx * 60}ms`,
            }}
          >
            {/* Connector line */}
            <div className="relative flex flex-col items-center">
              <div
                className={`w-5 h-5 rounded-full flex items-center justify-center flex-shrink-0 transition-all duration-300 ${
                  step.status === "done"
                    ? "bg-sunshine/20 border border-sunshine/40"
                    : step.status === "active"
                    ? "bg-sunshine/10 border border-sunshine/60 animate-pulse"
                    : step.status === "error"
                    ? "bg-hibiscus/20 border border-hibiscus/40"
                    : "bg-cream/5 border border-cream/10"
                }`}
              >
                {step.status === "pending" || step.status === "skipped" ? (
                  <span className="w-1.5 h-1.5 rounded-full bg-cream/20" />
                ) : (
                  STEP_ICON[step.status]
                )}
              </div>
              {/* Connector */}
              {idx < steps.length - 1 && (
                <div
                  className={`w-px mt-1 transition-all duration-500 ${
                    step.status === "done" ? "bg-sunshine/30 h-4" : "bg-cream/10 h-4"
                  }`}
                />
              )}
            </div>

            {/* Label */}
            <div
              className={`flex items-center gap-2 text-sm transition-all duration-300 ${
                step.status === "active"
                  ? "text-sunshine font-bold"
                  : step.status === "done"
                  ? "text-cream/80"
                  : step.status === "error"
                  ? "text-hibiscus"
                  : "text-muted-text/60"
              }`}
            >
              <span>{step.emoji}</span>
              <span className="tracking-wide">{step.label}</span>
              {step.status === "active" && (
                <span className="text-sunshine/60 text-xs animate-pulse">...</span>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// Default pipeline steps
export const DEFAULT_STEPS: PipelineStep[] = [
  { id: "listen", label: "Listening", emoji: "🎙", status: "pending" },
  { id: "transcribe", label: "Transcribing", emoji: "📝", status: "pending" },
  { id: "retrieve", label: "Retrieving", emoji: "🔎", status: "pending" },
  { id: "safety", label: "Safety Check", emoji: "🛡", status: "pending" },
  { id: "generate", label: "Generating", emoji: "🧠", status: "pending" },
  { id: "ground", label: "Grounding", emoji: "⚡", status: "pending" },
  { id: "done", label: "Answered", emoji: "✓", status: "pending" },
];
