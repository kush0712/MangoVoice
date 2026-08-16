"use client";

import React from "react";
import { Mic, Square, Loader2, ArrowUpRight } from "lucide-react";
import Waveform from "./Waveform";

export type RecordingState = "idle" | "listening" | "processing" | "done" | "error";

interface TaskStudioCardProps {
  state: RecordingState;
  onMicClick: () => void;
  onDemoClick: (queryText: string) => void;
  disabled?: boolean;
  amplitude?: number;
  transcript?: string | null;
  language?: string;
  onLanguageChange?: (lang: string) => void;
}

export default function TaskStudioCard({
  state,
  onMicClick,
  onDemoClick,
  disabled = false,
  amplitude = 0,
  transcript,
  language = "auto",
  onLanguageChange,
}: TaskStudioCardProps) {
  const isListening = state === "listening";
  const isProcessing = state === "processing";

  return (
    <div className="w-full card-cream p-6 md:p-10 relative z-10 transition-all duration-300">
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 md:gap-10 items-center">
        {/* ── Left Column: Interactive Mic Circle & Tilted Waveform Badge ── */}
        <div className="lg:col-span-5 flex flex-col items-center justify-center relative py-4">
          <div className="relative w-64 h-64 sm:w-72 sm:h-72 flex items-center justify-center">
            {/* Outer Orbiting Dashed Ring (Magenta) */}
            <div
              className="absolute inset-0 rounded-full pointer-events-none"
              style={{
                border: "3px dashed #FF0080",
                animation: isListening
                  ? "spin 5s linear infinite"
                  : isProcessing
                  ? "spin 2.5s linear infinite"
                  : "spin 20s linear infinite",
              }}
            />

            {/* Mango-Mic Accent Badge at Top-Right */}
            <div
              className="absolute top-2 right-4 z-20 w-11 h-11 rounded-full overflow-hidden flex items-center justify-center shadow-md select-none transition-transform hover:scale-110 border-2 border-sunshine/50 bg-forest-dark p-1"
              style={{
                boxShadow: "2px 3px 0px rgba(0,0,0,0.3)",
              }}
              title="MangoVoice"
            >
              <img
                src="/mango-emblem.png"
                alt="Mango"
                className="w-full h-full object-contain"
              />
            </div>

            {/* Pulsing Listening Aura */}
            {isListening && (
              <div
                className="absolute inset-4 rounded-full animate-ping opacity-30 pointer-events-none"
                style={{ backgroundColor: "#FF0080" }}
              />
            )}

            {/* Big Interactive Mic Button Circle */}
            <button
              id="main-voice-mic-btn"
              onClick={onMicClick}
              disabled={disabled || isProcessing}
              aria-label={
                isListening
                  ? "Stop recording"
                  : isProcessing
                  ? "Processing question"
                  : "Start recording voice question"
              }
              className="relative z-10 w-48 h-48 sm:w-52 sm:h-52 rounded-full flex flex-col items-center justify-center cursor-pointer transition-all duration-200 group select-none"
              style={{
                backgroundColor: isListening ? "#FF0080" : "#000000",
                border: "4px solid #FFFFFF",
                boxShadow: isListening
                  ? "0 0 30px rgba(255, 0, 128, 0.6), inset 0 0 20px rgba(0,0,0,0.2)"
                  : "0 8px 24px rgba(0, 0, 0, 0.4), inset 0 4px 10px rgba(255,255,255,0.15)",
                transform: isListening ? "scale(1.03)" : "scale(1)",
              }}
            >
              {/* Mic Icon / State Visualizer */}
              <div className="text-cream mb-2 transition-transform group-hover:scale-110">
                {isListening ? (
                  <Square size={44} strokeWidth={2.5} fill="#FFFFFF" className="animate-pulse" />
                ) : isProcessing ? (
                  <Loader2 size={48} strokeWidth={2.5} className="animate-spin text-sunshine" />
                ) : (
                  <Mic size={52} strokeWidth={2} />
                )}
              </div>

              {/* State Label */}
              <span
                className="font-hhg-mono font-bold text-xs sm:text-sm tracking-widest uppercase"
                style={{
                  color: isListening ? "#FFFFFF" : "#FFFF00",
                }}
              >
                {isListening
                  ? "STOP RECORDING"
                  : isProcessing
                  ? "TRANSCRIBING..."
                  : "PRESS TO SPEAK"}
              </span>

              {/* Live Waveform inside circle while recording */}
              {isListening && (
                <div className="mt-2 w-32">
                  <Waveform isActive={true} amplitude={amplitude} bars={16} />
                </div>
              )}
            </button>

            {/* Tilted White Audio ID Badge Card at Bottom-Left */}
            <div
              className="absolute -bottom-2 -left-2 sm:-bottom-4 sm:-left-4 z-20 bg-white px-4 py-2.5 rounded-xl shadow-lg border-2 border-ink/20 select-none pointer-events-none"
              style={{
                transform: "rotate(-7deg)",
                boxShadow: "4px 6px 0px rgba(0,0,0,0.25)",
              }}
            >
              {/* Green Waveform Bars Graphic */}
              <div className="flex items-end gap-1 h-5 mb-1.5">
                <span className="w-1 bg-forest h-2 rounded-full" />
                <span className="w-1 bg-forest h-4 rounded-full" />
                <span className="w-1 bg-forest h-5 rounded-full" />
                <span className="w-1 bg-forest h-3 rounded-full" />
                <span className="w-1 bg-forest h-4 rounded-full" />
                <span className="w-1 bg-forest h-2 rounded-full" />
              </div>
              <div className="text-[10px] font-hhg-mono font-bold text-hibiscus uppercase tracking-wider">
                {isListening ? "RECORDING AUDIO..." : "VOICE INPUT"}
              </div>
            </div>
          </div>

          {/* Language selector under mic */}
          <div className="mt-4 flex items-center gap-2">
            <span className="text-xs font-hhg-mono text-muted-text uppercase font-bold">Language:</span>
            <div className="flex gap-1">
              {[
                { id: "auto", label: "Auto (Codemix)" },
                { id: "hi-IN", label: "हिन्दी (Hindi)" },
                { id: "en-IN", label: "English" },
              ].map((l) => (
                <button
                  key={l.id}
                  onClick={() => onLanguageChange?.(l.id)}
                  disabled={isListening || isProcessing}
                  className={`text-[11px] font-hhg-mono px-2.5 py-1 rounded-tag transition-all cursor-pointer ${
                    language === l.id
                      ? "bg-forest text-cream font-bold shadow-sm"
                      : "bg-muted-fill/30 text-ink/70 hover:bg-muted-fill/60"
                  }`}
                >
                  {l.label}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* ── Right Column: MangoVoice Features & Specs ── */}
        <div className="lg:col-span-7 flex flex-col justify-between space-y-4">
          <div>
            {/* Eyebrow */}
            <div className="text-hibiscus font-hhg-mono font-bold text-xs uppercase tracking-[0.2em] mb-1">
              VOICE RAG ENGINE
            </div>

            {/* Title in condensed deep green serif */}
            <h2
              className="text-2xl sm:text-3xl md:text-4xl font-bold font-hhg-title leading-tight"
              style={{ color: "#000000" }}
            >
              Ask in Hindi, English or Hinglish
            </h2>

            {/* Description */}
            <p className="mt-3 text-xs sm:text-sm font-hhg-mono leading-relaxed text-ink/80">
              Speak your question and get a verifiable, strictly grounded answer. MangoVoice
              connects Sarvam Saaras v3 STT, LanceDB hybrid retrieval, and Groq Llama 3.1
              into a sub-200ms, hallucination-free pipeline.
            </p>
          </div>

          {/* Bullet Points with Pink Stars */}
          <div className="space-y-1.5 py-1">
            {[
              "Real speech-to-text input via Sarvam Saaras v3 with Indic codemix support",
              "Hybrid retrieval blending 384-dim multilingual vectors + BM25 via RRF",
              "Sub-200ms RAG core target for ultra-responsive voice interactions",
              "Groq Llama 3.1 tool-contract generation with strict refusal on low evidence",
              "4-Layer safety guardrails + sentence-level factual grounding verification",
              "Trained and benchmarked over ai4bharat/MSMARCO-XI multilingual dataset",
            ].map((bullet, idx) => (
              <div key={idx} className="flex items-start gap-2.5 text-xs font-hhg-mono text-ink/90">
                <span className="text-hibiscus text-sm leading-none flex-shrink-0 mt-0.5">✦</span>
                <span>{bullet}</span>
              </div>
            ))}
          </div>

          {/* System Status Box */}
          <div
            className="p-3 rounded-xl select-none"
            style={{
              backgroundColor: "rgba(255, 0, 128, 0.07)",
              border: "1.5px solid rgba(255, 0, 128, 0.35)",
            }}
          >
            <div className="text-xs font-hhg-mono font-bold text-hibiscus uppercase tracking-wider flex items-center justify-between flex-wrap gap-2">
              <span className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-hibiscus animate-ping" />
                PIPELINE READY · 5 CHUNKING STRATEGIES · MSMARCO-XI INDEX
              </span>
              <span className="opacity-80">FAST & GROUNDED</span>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex flex-wrap items-center gap-3 pt-2">
            {/* Primary Action Button */}
            <button
              id="task-action-speak-btn"
              onClick={onMicClick}
              disabled={disabled || isProcessing}
              className="btn-hhg-pink text-xs sm:text-sm"
            >
              {isListening ? (
                <>
                  <Square size={16} fill="currentColor" /> STOP & TRANSCRIBE
                </>
              ) : isProcessing ? (
                <>
                  <Loader2 size={16} className="animate-spin" /> RETRIEVING ANSWER...
                </>
              ) : (
                <>
                  <Mic size={16} /> START VOICE QUERY
                </>
              )}
            </button>

            {/* Outline Demo Button */}
            <button
              onClick={() => onDemoClick("What was the immediate impact of the Manhattan Project?")}
              disabled={isListening || isProcessing}
              className="btn-hhg-outline-pink text-xs sm:text-sm"
            >
              RUN DEMO QUERY <ArrowUpRight size={15} />
            </button>

            {/* Yellow Results Button */}
            <a
              href="#results-noticeboard"
              className="btn-hhg-yellow text-xs sm:text-sm hidden sm:inline-flex"
            >
              VIEW LIVE RESULTS
            </a>
          </div>
        </div>
      </div>
    </div>
  );
}
