"use client";

import React, { useState } from "react";
import { PushPin } from "./GoaIllustrations";
import { Volume2, VolumeX, ShieldAlert } from "lucide-react";
import { QueryResponse, Source } from "@/lib/api";

interface NoticeBoardResultsProps {
  result: QueryResponse | null;
  isLoading: boolean;
}

export default function NoticeBoardResults({ result, isLoading }: NoticeBoardResultsProps) {
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [expandedChunk, setExpandedChunk] = useState<number | null>(0);
  const audioRef = React.useRef<HTMLAudioElement | null>(null);

  if (!result && !isLoading) {
    return (
      <section id="results-noticeboard" className="py-12 relative z-10 text-center">
        <div className="text-sunshine font-hhg-mono font-bold text-xs uppercase tracking-[0.25em] mb-2">
          PINNED UP
        </div>
        <h3 className="text-3xl md:text-5xl font-hhg-fat text-cream shadow-hhg-text mb-8">
          NOTICE BOARD
        </h3>

        {/* Empty state pinned cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-5xl mx-auto px-4">
          <div className="card-pinned p-6 relative select-none" style={{ transform: "rotate(-2deg)" }}>
            <PushPin color="pink" />
            <div className="text-xs font-hhg-mono text-hibiscus uppercase font-bold mb-2 tracking-wider">
              VOICE INPUT
            </div>
            <p className="font-hhg-mono text-sm text-ink/70 leading-relaxed">
              Press the big microphone above to ask a question in Hindi, English, or Hinglish.
            </p>
            <div className="mt-4">
              <span className="btn-hhg-pink text-[11px] py-1 px-4 cursor-default">
                AWAITING AUDIO
              </span>
            </div>
          </div>

          <div className="card-pinned p-6 relative select-none" style={{ transform: "rotate(1deg)" }}>
            <PushPin color="yellow" />
            <div className="text-xs font-hhg-mono text-forest uppercase font-bold mb-2 tracking-wider">
              GROUNDED ANSWER
            </div>
            <p className="font-hhg-mono text-sm text-ink/70 leading-relaxed">
              Answers generated strictly from retrieved evidence with tool-contract enforcement.
            </p>
            <div className="mt-4">
              <span className="btn-hhg-yellow text-[11px] py-1 px-4 cursor-default">
                ZERO HALLUCINATIONS
              </span>
            </div>
          </div>

          <div className="card-pinned p-6 relative select-none" style={{ transform: "rotate(2deg)" }}>
            <PushPin color="pink" />
            <div className="text-xs font-hhg-mono text-hibiscus uppercase font-bold mb-2 tracking-wider">
              EVIDENCE SOURCES
            </div>
            <p className="font-hhg-mono text-sm text-ink/70 leading-relaxed">
              Inspect top chunks from LanceDB (dense + BM25) and verification scores.
            </p>
            <div className="mt-4">
              <span className="btn-hhg-pink text-[11px] py-1 px-4 cursor-default">
                MSMARCO-XI INDEX
              </span>
            </div>
          </div>
        </div>
      </section>
    );
  }

  const isRefusal = result?.confidence === "refused" || !result?.answer;

  const handleSpeak = async () => {
    if (!result?.answer) return;

    // If currently speaking, stop immediately
    if (isSpeaking) {
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current.currentTime = 0;
        audioRef.current = null;
      }
      if (typeof window !== "undefined" && window.speechSynthesis) {
        window.speechSynthesis.cancel();
      }
      setIsSpeaking(false);
      return;
    }

    setIsSpeaking(true);

    // 1. Try Sarvam AI high-quality Indian TTS
    try {
      const resp = await fetch("/api/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: result.answer,
          language: result.language || "auto",
        }),
      });

      if (resp.ok) {
        const data = await resp.json();
        if (data.audio_base64) {
          const audio = new Audio(`data:audio/wav;base64,${data.audio_base64}`);
          audioRef.current = audio;
          audio.onended = () => {
            setIsSpeaking(false);
            audioRef.current = null;
          };
          audio.onerror = () => {
            setIsSpeaking(false);
            audioRef.current = null;
          };
          await audio.play();
          return;
        }
      }
    } catch (e) {
      console.warn("Sarvam TTS fallback to Web Speech API", e);
    }

    // 2. Fallback to browser Web Speech API with Devanagari / Hindi detection
    if (typeof window !== "undefined" && window.speechSynthesis) {
      window.speechSynthesis.cancel();
      const hasDevanagari = /[\u0900-\u097F]/.test(result.answer);
      const utterance = new SpeechSynthesisUtterance(result.answer);
      utterance.lang = hasDevanagari ? "hi-IN" : "en-IN";
      utterance.rate = hasDevanagari ? 0.88 : 0.95;

      const voices = window.speechSynthesis.getVoices();
      const matchedVoice = voices.find((v) =>
        hasDevanagari
          ? v.lang.startsWith("hi") || v.name.toLowerCase().includes("hindi")
          : v.lang.startsWith("en-IN") || v.lang.startsWith("en")
      );
      if (matchedVoice) {
        utterance.voice = matchedVoice;
      }

      utterance.onend = () => setIsSpeaking(false);
      utterance.onerror = () => setIsSpeaking(false);

      // Store utterance on window to prevent premature browser garbage collection
      (window as unknown as { _currentSpeechUtterance?: SpeechSynthesisUtterance })._currentSpeechUtterance = utterance;
      window.speechSynthesis.speak(utterance);
    } else {
      setIsSpeaking(false);
    }
  };

  return (
    <section id="results-noticeboard" className="py-12 relative z-10">
      <div className="text-center mb-8">
        <div className="text-sunshine font-hhg-mono font-bold text-xs uppercase tracking-[0.25em] mb-2">
          PINNED UP · LIVE PIPELINE EVIDENCE
        </div>
        <h3 className="text-3xl md:text-5xl font-hhg-fat text-cream shadow-hhg-text">
          NOTICE BOARD
        </h3>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 max-w-6xl mx-auto px-4">
        {/* ── Card 1: Your Question ── */}
        <div className="lg:col-span-4 flex flex-col">
          <div
            className="card-pinned p-6 relative flex-1 flex flex-col justify-between"
            style={{ transform: "rotate(-1.5deg)" }}
          >
            <PushPin color="pink" />
            <div>
              <div className="flex items-center justify-between mb-3 border-b border-muted-fill/40 pb-2">
                <span className="text-xs font-hhg-mono text-hibiscus uppercase font-bold tracking-wider">
                  QUESTION TRANSCRIPT
                </span>
                {result?.language && (
                  <span className="text-[10px] font-hhg-mono bg-forest text-cream px-2 py-0.5 rounded-full font-bold uppercase">
                    {result.language}
                  </span>
                )}
              </div>

              <blockquote className="font-hhg-mono text-sm text-ink font-bold leading-relaxed my-3 italic">
                &ldquo;{result?.transcript || "..."}&rdquo;
              </blockquote>
            </div>

            <div className="mt-4 pt-3 border-t border-muted-fill/40 text-[11px] font-hhg-mono text-muted-text flex items-center justify-between">
              <span>Sarvam STT Latency:</span>
              <span className="font-bold text-forest">
                {result?.latency?.stt_ms ? `${result.latency.stt_ms.toFixed(0)} ms` : "0 ms (Text)"}
              </span>
            </div>
          </div>
        </div>

        {/* ── Card 2: Grounded Answer & Grounding Meter ── */}
        <div className="lg:col-span-5 flex flex-col">
          <div className="card-pinned p-6 relative flex-1 flex flex-col justify-between" style={{ transform: "rotate(0.5deg)" }}>
            <PushPin color="yellow" />
            <div>
              <div className="flex items-center justify-between mb-3 border-b border-muted-fill/40 pb-2">
                <span className="text-xs font-hhg-mono text-forest uppercase font-bold tracking-wider">
                  GROUNDED ANSWER
                </span>
                <div className="flex items-center gap-2">
                  <span
                    className={`text-[11px] font-hhg-mono font-bold px-2.5 py-0.5 rounded-tag uppercase ${
                      result?.confidence === "high"
                        ? "bg-sunshine text-forest border border-forest/20"
                        : result?.confidence === "medium"
                        ? "bg-lime text-forest border border-forest/20"
                        : "bg-hibiscus text-cream"
                    }`}
                  >
                    {result?.confidence || "REFUSED"} CONFIDENCE
                  </span>
                </div>
              </div>

              {isRefusal ? (
                <div className="bg-hibiscus/10 border border-hibiscus/30 rounded-xl p-4 my-3 text-ink">
                  <div className="flex items-center gap-2 text-hibiscus font-bold text-xs uppercase mb-1">
                    <ShieldAlert size={16} /> Refusal Activated
                  </div>
                  <p className="font-hhg-mono text-xs leading-relaxed">
                    {result?.refusal_message || "Insufficient evidence in MSMARCO-XI index to provide a confident, grounded response."}
                  </p>
                </div>
              ) : (
                <p className="font-hhg-mono text-sm text-ink leading-relaxed my-3 font-medium">
                  {result?.answer}
                </p>
              )}
            </div>

            {/* Grounding & Audio Controls */}
            <div className="mt-4 pt-3 border-t border-muted-fill/40 space-y-3">
              {result?.grounding_score !== null && result?.grounding_score !== undefined && (
                <div>
                  <div className="flex justify-between text-[11px] font-hhg-mono text-muted-text mb-1">
                    <span>Factual Grounding Score:</span>
                    <span className="font-bold text-forest">
                      {Math.round(result.grounding_score * 100)}%
                    </span>
                  </div>
                  <div className="w-full h-2 bg-muted-fill/30 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-forest transition-all duration-700"
                      style={{ width: `${Math.round(result.grounding_score * 100)}%` }}
                    />
                  </div>
                </div>
              )}

              <div className="flex items-center justify-between pt-1">
                <button
                  onClick={handleSpeak}
                  disabled={!result?.answer}
                  className="btn-hhg-pink text-xs py-1.5 px-4 disabled:opacity-40"
                >
                  {isSpeaking ? (
                    <>
                      <VolumeX size={14} /> STOP AUDIO
                    </>
                  ) : (
                    <>
                      <Volume2 size={14} /> READ ANSWER ALOUD
                    </>
                  )}
                </button>

                <div className="text-[11px] font-hhg-mono text-muted-text">
                  RAG Core: <span className="font-bold text-forest">{result?.latency?.rag_core_ms?.toFixed(0)} ms</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* ── Card 3: Retrieved Evidence Chunks ── */}
        <div className="lg:col-span-3 flex flex-col">
          <div
            className="card-pinned p-5 relative flex-1 flex flex-col"
            style={{ transform: "rotate(1.5deg)" }}
          >
            <PushPin color="green" />
            <div className="flex items-center justify-between mb-3 border-b border-muted-fill/40 pb-2">
              <span className="text-xs font-hhg-mono text-forest-light uppercase font-bold tracking-wider">
                EVIDENCE ({result?.sources?.length || 0})
              </span>
              <span className="text-[10px] font-hhg-mono bg-sunshine text-forest px-2 py-0.5 rounded font-bold">
                LANCE DB
              </span>
            </div>

            {/* Sources Accordion */}
            <div className="space-y-2 overflow-y-auto max-h-[300px] pr-1">
              {result?.sources?.map((src: Source, idx: number) => {
                const isExpanded = expandedChunk === idx;
                return (
                  <div
                    key={src.chunk_id}
                    className="border border-muted-fill/60 rounded-xl p-2.5 bg-cream/70 text-ink text-xs transition-all hover:bg-cream"
                  >
                    <div
                      className="flex items-center justify-between cursor-pointer"
                      onClick={() => setExpandedChunk(isExpanded ? null : idx)}
                    >
                      <div className="flex items-center gap-1.5 font-bold font-hhg-mono text-[11px]">
                        <span className="w-4 h-4 rounded-full bg-forest text-cream flex items-center justify-center text-[9px]">
                          {idx + 1}
                        </span>
                        <span className="truncate max-w-[120px]">{src.chunk_id}</span>
                      </div>
                      <span className="text-[10px] font-mono text-muted-text">
                        {(src.rrf_score ?? src.score).toFixed(3)}
                      </span>
                    </div>

                    {isExpanded && (
                      <p className="mt-2 text-[11px] font-hhg-mono text-ink/80 leading-relaxed border-t border-muted-fill/40 pt-2">
                        {src.text}
                      </p>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
