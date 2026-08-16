"use client";

import React, { useCallback, useEffect, useState } from "react";
import {
  GoaTextileRibbon,
  GoaLeftPalmTree,
  GoaRightPalmTree,
  GoaRisingSun,
  DevanagariOverlay,
} from "@/components/GoaIllustrations";
import TaskStudioCard, { RecordingState } from "@/components/TaskStudioCard";
import NoticeBoardResults from "@/components/NoticeBoardResults";
import GoaPipelineAndStrategies from "@/components/GoaPipelineAndStrategies";
import GoaFaqAndFooter from "@/components/GoaFaqAndFooter";
import { DEFAULT_STEPS, PipelineStep } from "@/components/PipelineStatus";
import { useVoiceRecorder } from "@/lib/useVoiceRecorder";
import { queryAudio, queryText, checkHealth, QueryResponse } from "@/lib/api";

function updateStep(
  steps: PipelineStep[],
  id: string,
  status: PipelineStep["status"]
): PipelineStep[] {
  return steps.map((s) => (s.id === id ? { ...s, status } : s));
}

function activateStep(steps: PipelineStep[], id: string): PipelineStep[] {
  return steps.map((s) => {
    if (s.id === id) return { ...s, status: "active" };
    if (s.status === "active") return { ...s, status: "done" };
    return s;
  });
}

export default function Home() {
  const [appState, setAppState] = useState<RecordingState>("idle");
  const [steps, setSteps] = useState<PipelineStep[]>(DEFAULT_STEPS);
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [backendStatus, setBackendStatus] = useState<"checking" | "ready" | "degraded" | "offline">("checking");
  const [amplitude, setAmplitude] = useState(0);
  const [language, setLanguage] = useState("auto");

  const {
    isRecording,
    start: startRecording,
    stop: stopRecording,
    error: recError,
    isSupported,
  } = useVoiceRecorder({
    maxSeconds: 25,
    silenceTimeout: 2500,
    onAmplitude: setAmplitude,
  });

  // ── Backend health check ──────────────────────────────────────────────────
  useEffect(() => {
    const check = async () => {
      try {
        const h = await checkHealth();
        setBackendStatus(h.status === "ready" ? "ready" : "degraded");
      } catch {
        setBackendStatus("offline");
      }
    };
    check();
    const interval = setInterval(check, 30000);
    return () => clearInterval(interval);
  }, []);

  // ── Process query result ──────────────────────────────────────────────────
  const handleResult = useCallback((res: QueryResponse) => {
    setResult(res);
    setSteps((prev) => {
      let s = prev;
      if (res.status === "answered") {
        s = activateStep(s, "done");
        s = updateStep(s, "done", "done");
        s = s.map((step) =>
          ["safety", "generate", "ground"].includes(step.id)
            ? { ...step, status: "done" }
            : step
        );
      } else {
        s = s.map((step) =>
          step.status === "active"
            ? { ...step, status: "done" }
            : step.status === "pending"
            ? { ...step, status: "skipped" }
            : step
        );
      }
      return s;
    });
    setAppState("done");

    // Scroll smoothly to noticeboard results
    setTimeout(() => {
      const el = document.getElementById("results-noticeboard");
      if (el) el.scrollIntoView({ behavior: "smooth" });
    }, 200);
  }, []);

  // ── Handle mic button click ───────────────────────────────────────────────
  const handleMicClick = useCallback(async () => {
    if (appState === "processing") return;

    if (appState === "listening") {
      setAppState("processing");
      setSteps((prev) => updateStep(activateStep(prev, "transcribe"), "listen", "done"));

      try {
        const blob = await stopRecording();
        if (!blob) throw new Error("No audio recorded");

        setSteps((prev) => activateStep(prev, "retrieve"));
        const res = await queryAudio(blob, language);
        handleResult(res);
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : "Query processing failed";
        setErrorMsg(msg);
        setAppState("error");
      }
      return;
    }

    // Start recording
    setAppState("listening");
    setResult(null);
    setErrorMsg(null);
    setSteps(DEFAULT_STEPS.map((s) => ({ ...s, status: "pending" as const })));
    setSteps((prev) => activateStep(prev, "listen"));

    try {
      await startRecording();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Microphone access denied";
      setErrorMsg(msg);
      setAppState("error");
    }
  }, [appState, startRecording, stopRecording, language, handleResult]);


  // ── Handle demo query click ───────────────────────────────────────────────
  const handleDemoQuery = useCallback(
    async (text: string) => {
      if (appState === "processing" || appState === "listening") return;

      setAppState("processing");
      setResult(null);
      setErrorMsg(null);
      setSteps(DEFAULT_STEPS.map((s) => ({ ...s, status: "pending" as const })));
      setSteps((prev) => activateStep(prev, "transcribe"));

      try {
        setSteps((prev) => activateStep(prev, "retrieve"));
        const res = await queryText(text, language);
        handleResult(res);
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : "Query failed";
        setErrorMsg(msg);
        setAppState("error");
      }
    },
    [appState, language, handleResult]
  );

  return (
    <div className="relative min-h-screen bg-forest text-cream overflow-x-hidden">
      {/* ── Top Goan Geometric Textile Ribbon ── */}
      <GoaTextileRibbon />

      {/* ── Left & Right Framing Palm Trees ── */}
      <div className="hidden 2xl:block absolute top-12 left-0 w-44 z-0 pointer-events-none opacity-90">
        <GoaLeftPalmTree />
      </div>
      <div className="hidden 2xl:block absolute top-12 right-0 w-44 z-0 pointer-events-none opacity-90">
        <GoaRightPalmTree />
      </div>

      {/* ── Header Navigation Bar ── */}
      <header className="relative z-20 max-w-6xl mx-auto px-6 py-6 flex items-center justify-between">
        {/* Left: MangoVoice Brand Mark */}
        <div className="flex items-center gap-2.5">
          <img
            src="/mango-emblem.png"
            alt="Mango"
            className="w-10 h-10 sm:w-11 sm:h-11 object-contain drop-shadow-sm select-none transition-transform hover:scale-105"
          />
          <div>
            <div className="font-hhg-fat text-2xl md:text-3xl text-sunshine leading-none tracking-tight">
              Mango<span className="text-cream">Voice</span>
            </div>
            <div className="font-hhg-mono text-[10px] text-cream/70 uppercase tracking-widest">
              MULTILINGUAL INDIC RAG
            </div>
          </div>
        </div>

        {/* Center / Right: Nav Links + Signature Textile SPEAK NOW Button */}
        <div className="flex items-center gap-6 md:gap-8">
          <nav className="hidden md:flex items-center gap-6 font-hhg-mono text-xs uppercase tracking-wider text-cream/80">
            <a href="#studio-section" className="hover:text-sunshine transition-colors">
              STUDIO
            </a>
            <a href="#results-noticeboard" className="hover:text-sunshine transition-colors">
              NOTICE BOARD
            </a>
            <a href="#faqs-section" className="hover:text-sunshine transition-colors">
              FAQS
            </a>
          </nav>

          {/* Backend Status Indicator */}
          <div className="hidden sm:flex items-center gap-2 bg-forest-dark/80 px-3 py-1.5 rounded-full border border-cream/20 text-xs font-hhg-mono">
            <span
              className={`w-2 h-2 rounded-full ${
                backendStatus === "ready"
                  ? "bg-sunshine"
                  : backendStatus === "degraded"
                  ? "bg-sunshine/60"
                  : "bg-hibiscus"
              }`}
            />
            <span className="text-[11px] text-cream/80 uppercase">
              {backendStatus === "ready"
                ? "INDEX READY"
                : backendStatus === "degraded"
                ? "DEGRADED"
                : backendStatus === "checking"
                ? "CONNECTING"
                : "OFFLINE"}
            </span>
          </div>

          {/* Signature Textile Button */}
          <button
            onClick={handleMicClick}
            disabled={appState === "processing"}
            className="btn-hhg-textile text-sm"
          >
            {appState === "listening" ? "STOP MIC" : "SPEAK NOW"}
          </button>
        </div>
      </header>

      {/* ── Main Hero Section ── */}
      <section className="relative z-10 pt-6 pb-12 text-center max-w-5xl mx-auto px-4">
        {/* Rising Sun Illustration */}
        <div className="absolute top-0 left-1/2 -translate-x-1/2 -translate-y-8 pointer-events-none opacity-40 z-0">
          <GoaRisingSun />
        </div>

        {/* Main Title with Devanagari Script Overlay */}
        <div className="relative z-10 inline-block my-4">
          <h1
            className="font-hhg-title font-black text-cream uppercase tracking-tight select-none"
            style={{
              fontSize: "clamp(3.2rem, 10vw, 7.5rem)",
              lineHeight: 0.95,
              textShadow: "4px 6px 0px #000000, 8px 12px 0px rgba(0,0,0,0.3)",
              color: "#FFFFFF",
            }}
          >
            MANGO VOICE
          </h1>

          {/* Iconic Devanagari Magenta Script Overlay Badge */}
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-20">
            <DevanagariOverlay text="मैंगो" subText="आवाज़" />
          </div>
        </div>

        {/* Hero Product Subtitle */}
        <div className="flex items-center justify-between max-w-2xl mx-auto mt-4 px-4 text-xs sm:text-sm font-hhg-mono text-sunshine font-bold uppercase tracking-widest select-none">
          <span>VOICE-ENABLED RAG</span>
          <span>·</span>
          <span>HINDI · ENGLISH · HINGLISH</span>
          <span>·</span>
          <span>MSMARCO-XI</span>
        </div>
      </section>

      {/* ── Section: The Interactive Voice Studio Card ── */}
      <section id="studio-section" className="max-w-5xl mx-auto px-4 py-6 relative z-10">
        {/* Error notification banner if any */}
        {(errorMsg || recError) && (
          <div className="mb-6 p-4 rounded-card bg-hibiscus text-cream font-hhg-mono text-xs sm:text-sm font-bold shadow-hhg-btn flex items-center justify-between">
            <span>⚠ {errorMsg || recError}</span>
            <button
              onClick={() => setErrorMsg(null)}
              className="underline text-sunshine ml-4 uppercase text-xs cursor-pointer"
            >
              DISMISS
            </button>
          </div>
        )}

        <TaskStudioCard
          state={appState}
          onMicClick={handleMicClick}
          onDemoClick={handleDemoQuery}
          disabled={!isSupported && backendStatus === "offline"}
          amplitude={amplitude}
          language={language}
          onLanguageChange={setLanguage}
        />
      </section>

      {/* ── Section: Pinned Notice Board Results ── */}
      <NoticeBoardResults result={result} isLoading={appState === "processing"} />

      {/* ── Section: Pipeline Chevrons + Presets + 5 Strategies ── */}
      <GoaPipelineAndStrategies
        steps={steps}
        onSelectDemoQuery={handleDemoQuery}
        isProcessing={appState === "processing" || appState === "listening"}
      />

      {/* ── Section: FAQs & Illustrated Beach Footer ── */}
      <GoaFaqAndFooter />
    </div>
  );
}
