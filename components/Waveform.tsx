"use client";

import { useEffect, useRef } from "react";

interface WaveformProps {
  isActive: boolean;
  amplitude?: number; // 0–1
  bars?: number;
}

export default function Waveform({ isActive, amplitude = 0.5, bars = 20 }: WaveformProps) {
  const barsRef = useRef<(HTMLDivElement | null)[]>([]);

  useEffect(() => {
    if (!isActive) {
      barsRef.current.forEach((bar) => {
        if (bar) bar.style.transform = "scaleY(0.15)";
      });
      return;
    }

    const interval = setInterval(() => {
      barsRef.current.forEach((bar, i) => {
        if (!bar) return;
        // Natural waveform shape: higher in center, smaller at edges
        const distFromCenter = Math.abs(i - bars / 2) / (bars / 2);
        const envelope = 1 - distFromCenter * 0.6;
        const randomVariance = 0.3 + Math.random() * 0.7;
        const scale = Math.max(0.1, envelope * randomVariance * amplitude * 1.2);
        bar.style.transform = `scaleY(${scale})`;
        bar.style.transition = `transform ${80 + Math.random() * 60}ms ease`;
      });
    }, 80);

    return () => clearInterval(interval);
  }, [isActive, amplitude, bars]);

  return (
    <div className="flex items-center justify-center gap-[3px] h-12" aria-hidden="true">
      {Array.from({ length: bars }).map((_, i) => (
        <div
          key={i}
          ref={(el) => { barsRef.current[i] = el; }}
          className="rounded-full flex-shrink-0"
          style={{
            width: "4px",
            height: "40px",
            backgroundColor: isActive ? "#FFFF00" : "rgba(255, 255, 0,0.25)",
            transform: "scaleY(0.15)",
            transformOrigin: "center",
            transition: "transform 120ms ease, background-color 300ms ease",
          }}
        />
      ))}
    </div>
  );
}
