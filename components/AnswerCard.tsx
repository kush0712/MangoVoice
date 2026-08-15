"use client";

import { useState } from "react";
import { Volume2, VolumeX, AlertTriangle } from "lucide-react";
import ConfidenceBadge from "./ConfidenceBadge";

interface AnswerCardProps {
  answer: string | null;
  confidence: "high" | "medium" | "refused";
  confidenceScore?: number;
  refusalMessage?: string | null;
  groundingScore?: number;
  visible: boolean;
}

export default function AnswerCard({
  answer,
  confidence,
  confidenceScore,
  refusalMessage,
  groundingScore,
  visible,
}: AnswerCardProps) {
  const [isSpeaking, setIsSpeaking] = useState(false);

  if (!visible) return null;

  const isRefusal = confidence === "refused" || !answer;

  const handleSpeak = () => {
    if (!answer) return;
    if (isSpeaking) {
      window.speechSynthesis.cancel();
      setIsSpeaking(false);
      return;
    }
    const utterance = new SpeechSynthesisUtterance(answer);
    utterance.lang = "en-IN";
    utterance.rate = 0.9;
    utterance.onend = () => setIsSpeaking(false);
    utterance.onerror = () => setIsSpeaking(false);
    window.speechSynthesis.speak(utterance);
    setIsSpeaking(true);
  };

  return (
    <div className="animate-slide-up" style={{ animationDelay: "100ms" }}>
      {isRefusal ? (
        /* Refusal state */
        <div className="card-glass rounded-card p-6 border border-hibiscus/20">
          <div className="flex items-start gap-3">
            <AlertTriangle size={20} className="text-hibiscus flex-shrink-0 mt-0.5" />
            <div>
              <div className="eyebrow text-hibiscus mb-2">Not answered</div>
              <p className="text-cream/80 text-sm leading-relaxed font-mono">
                {refusalMessage || "I couldn't find enough evidence to answer that confidently."}
              </p>
              <div className="mt-3">
                <ConfidenceBadge level="refused" />
              </div>
            </div>
          </div>
        </div>
      ) : (
        /* Answer state */
        <div className="card-glass rounded-card p-6">
          {/* Header row */}
          <div className="flex items-center justify-between mb-4">
            <div className="eyebrow text-cream/60">Answer</div>
            <div className="flex items-center gap-3">
              <ConfidenceBadge
                level={confidence}
                score={confidenceScore}
                showScore={true}
              />
              {/* Read aloud button */}
              <button
                onClick={handleSpeak}
                id="read-aloud-btn"
                title={isSpeaking ? "Stop reading" : "Read answer aloud"}
                className={`flex items-center gap-1.5 text-xs font-mono uppercase tracking-wider transition-all px-3 py-1.5 rounded-pill border ${
                  isSpeaking
                    ? "border-hibiscus text-hibiscus"
                    : "border-cream/20 text-cream/50 hover:border-cream/40 hover:text-cream/80"
                }`}
              >
                {isSpeaking ? <VolumeX size={12} /> : <Volume2 size={12} />}
                {isSpeaking ? "Stop" : "Read"}
              </button>
            </div>
          </div>

          {/* Answer text */}
          <p className="text-cream text-base leading-relaxed font-mono whitespace-pre-wrap">
            {answer}
          </p>

          {/* Grounding indicator */}
          {groundingScore !== undefined && (
            <div className="mt-4 pt-4 border-t border-cream/10">
              <div className="flex items-center gap-2 text-xs font-mono text-muted-text">
                <span>Grounding score:</span>
                <div className="flex-1 h-1 bg-cream/10 rounded-full max-w-24">
                  <div
                    className="h-1 rounded-full bg-sunshine transition-all duration-700"
                    style={{ width: `${Math.round(groundingScore * 100)}%` }}
                  />
                </div>
                <span className="text-sunshine">{Math.round(groundingScore * 100)}%</span>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
