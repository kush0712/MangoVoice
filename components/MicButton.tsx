"use client";

import { Mic, Square, Loader2 } from "lucide-react";

export type MicState = "idle" | "listening" | "processing" | "error";

interface MicButtonProps {
  state: MicState;
  onClick: () => void;
  disabled?: boolean;
}

const STATE_CONFIG = {
  idle: {
    label: "SPEAK",
    subLabel: "Press to ask",
    bgClass: "bg-hibiscus hover:bg-[#D10074] active:scale-95",
    ringColor: "rgba(255, 0, 128,0.35)",
    icon: <Mic size={32} strokeWidth={2} />,
    ping: false,
  },
  listening: {
    label: "LISTENING",
    subLabel: "Press to stop",
    bgClass: "bg-sunshine hover:bg-sunshine active:scale-95",
    textClass: "text-forest",
    ringColor: "rgba(255, 255, 0,0.45)",
    icon: <Square size={28} strokeWidth={2.5} fill="currentColor" />,
    ping: true,
  },
  processing: {
    label: "PROCESSING",
    subLabel: "Thinking...",
    bgClass: "bg-forest-deep cursor-not-allowed",
    ringColor: "rgba(44,102,61,0.3)",
    icon: <Loader2 size={32} strokeWidth={2} className="animate-spin" />,
    ping: false,
  },
  error: {
    label: "TRY AGAIN",
    subLabel: "Error occurred",
    bgClass: "bg-hibiscus hover:bg-[#D10074] active:scale-95",
    ringColor: "rgba(255, 0, 128,0.35)",
    icon: <Mic size={32} strokeWidth={2} />,
    ping: false,
  },
};

export default function MicButton({ state, onClick, disabled }: MicButtonProps) {
  const cfg = STATE_CONFIG[state];
  const isListening = state === "listening";
  const textColor = isListening ? "text-forest" : "text-white";

  return (
    <div className="flex flex-col items-center gap-5">
      {/* Button */}
      <div className="relative">
        {/* Dashed rotating ring (halo) */}
        <div
          className="absolute inset-0 rounded-full pointer-events-none"
          style={{
            margin: "-12px",
            border: `2px dashed ${cfg.ringColor}`,
            borderRadius: "50%",
            animation: isListening
              ? "spin 6s linear infinite"
              : state === "idle"
              ? "spin 12s linear infinite"
              : "none",
          }}
        />

        {/* Ping animation for listening state */}
        {cfg.ping && (
          <div
            className="absolute inset-0 rounded-full animate-ping"
            style={{
              margin: "-4px",
              backgroundColor: "rgba(255, 255, 0,0.25)",
              borderRadius: "50%",
            }}
          />
        )}

        <button
          onClick={onClick}
          disabled={disabled || state === "processing"}
          aria-label={`${cfg.label} — ${cfg.subLabel}`}
          id="mic-button"
          className={`
            relative z-10 flex items-center justify-center
            w-24 h-24 rounded-full ${cfg.bgClass} ${textColor}
            shadow-2xl transition-all duration-200 select-none
          `}
          style={{
            boxShadow: `0 0 0 0 ${cfg.ringColor}, 0 8px 32px rgba(0,0,0,0.3)`,
          }}
        >
          {cfg.icon}
        </button>
      </div>

      {/* Labels */}
      <div className="text-center">
        <div className={`font-bold text-sm tracking-[0.15em] uppercase ${textColor === "text-forest" ? "text-sunshine" : "text-cream"}`}>
          {cfg.label}
        </div>
        <div className="text-muted-text text-xs tracking-wider mt-0.5">
          {cfg.subLabel}
        </div>
      </div>
    </div>
  );
}
