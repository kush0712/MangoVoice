import React from "react";

// ── Top Geometric Textile Ribbon (Screenshot 1 & 7) ──────────────────────────
export function GoaTextileRibbon() {
  return (
    <div className="w-full overflow-hidden bg-[#000000] border-b-2 border-sunshine select-none">
      <svg
        className="w-full h-7 text-sunshine block"
        preserveAspectRatio="repeat-x"
        viewBox="0 0 400 24"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        <pattern id="goa-textile-pattern" width="80" height="24" patternUnits="userSpaceOnUse">
          <polygon points="12,12 20,4 28,12 20,20" fill="#9333EA" />
          <circle cx="20" cy="12" r="2.5" fill="#FEFBEA" />
          <polygon points="52,12 60,4 68,12 60,20" fill="#8FC93A" />
          <circle cx="60" cy="12" r="2.5" fill="#FEFBEA" />
          <path d="M0 12h8 M32 12h16 M72 12h8" stroke="#FFFFFF" strokeWidth="1.5" />
          <circle cx="40" cy="8" r="1.5" fill="#FFFFFF" />
          <circle cx="40" cy="16" r="1.5" fill="#FFFFFF" />
          <polygon points="0,0 80,0 80,2 0,2" fill="#FFFFFF" />
          <polygon points="0,22 80,22 80,24 0,24" fill="#FFFFFF" />
        </pattern>
        <rect width="100%" height="24" fill="url(#goa-textile-pattern)" />
      </svg>
    </div>
  );
}

// ── Floral Cross Divider (Screenshot 10) ─────────────────────────────────────
export function GoaFloralDivider({ className = "" }: { className?: string }) {
  return (
    <div className={`flex items-center justify-center gap-4 py-4 select-none ${className}`}>
      <span className="text-forest-light/60 text-lg">✤</span>
      <span className="text-forest-light/80 text-xl">✤</span>
      <span className="text-sunshine text-2xl font-bold">✤</span>
      <span className="text-hibiscus text-3xl font-bold">✤</span>
      <span className="text-sunshine text-2xl font-bold">✤</span>
      <span className="text-forest-light/80 text-xl">✤</span>
      <span className="text-forest-light/60 text-lg">✤</span>
    </div>
  );
}

// ── Devanagari Magenta Script Overlay Badge (Screenshot 1, 8, 13) ──────────────
export function DevanagariOverlay({
  text = "मैंगो",
  subText = "आवाज़",
}: {
  text?: string;
  subText?: string;
}) {
  return (
    <div
      className="inline-flex items-center gap-2 px-5 py-1.5 rounded-full shadow-lg border-2 border-white select-none transition-transform hover:scale-105"
      style={{
        backgroundColor: "#9333EA",
        boxShadow: "0 0 20px rgba(147, 51, 234, 0.7), 3px 4px 0px rgba(0,0,0,0.3)",
      }}
    >
      <span className="font-hhg-devanagari text-2xl sm:text-3xl text-white font-bold tracking-wide">
        {text}
      </span>
      {subText && (
        <>
          <span className="text-white/60 text-xs font-mono">|</span>
          <span className="font-hhg-devanagari text-lg sm:text-xl text-sunshine font-bold">
            {subText}
          </span>
        </>
      )}
    </div>
  );
}

// ── 3D Push Pin for Notice Board (Screenshot 3) ──────────────────────────────
export function PushPin({ color = "pink" }: { color?: "pink" | "yellow" | "green" }) {
  const bg = color === "pink" ? "#9333EA" : color === "yellow" ? "#FFFFFF" : "#8FC93A";
  const border = color === "pink" ? "#7E22CE" : color === "yellow" ? "#FFFFFF" : "#000000";

  return (
    <div className="absolute -top-3.5 left-1/2 -translate-x-1/2 z-20 flex flex-col items-center select-none pointer-events-none">
      <div
        className="w-7 h-7 rounded-full flex items-center justify-center shadow-md"
        style={{
          backgroundColor: bg,
          border: `2px solid ${border}`,
          boxShadow: "0 4px 6px rgba(0,0,0,0.35), inset 0 2px 4px rgba(255,255,255,0.4)",
        }}
      >
        <div className="w-2 h-2 rounded-full bg-white shadow-inner" />
      </div>
      <div className="w-0.5 h-2 bg-neutral-400 -mt-0.5" />
    </div>
  );
}

// ── Left Framing Palm Tree (Screenshot 1 & 13) ────────────────────────────────
export function GoaLeftPalmTree() {
  return (
    <svg viewBox="0 0 160 380" fill="none" xmlns="http://www.w3.org/2000/svg" className="w-full h-auto">
      <path
        d="M-20 380 Q 20 220 50 140 Q 65 90 70 60"
        stroke="#000000"
        strokeWidth="18"
        strokeLinecap="round"
      />
      <path
        d="M-20 380 Q 20 220 50 140 Q 65 90 70 60"
        stroke="#000000"
        strokeWidth="10"
        strokeLinecap="round"
      />
      <path
        d="M70 60 Q 110 30 150 50 Q 120 75 70 60"
        fill="#000000"
        stroke="#000000"
        strokeWidth="2"
      />
      <path
        d="M70 60 Q 95 10 140 -5 Q 115 35 70 60"
        fill="#8FC93A"
        stroke="#000000"
        strokeWidth="2"
      />
      <path
        d="M70 60 Q 50 -10 60 -50 Q 40 -5 70 60"
        fill="#000000"
        stroke="#000000"
        strokeWidth="2"
      />
      <path
        d="M70 60 Q 20 15 -20 5 Q 15 45 70 60"
        fill="#8FC93A"
        stroke="#000000"
        strokeWidth="2"
      />
      <path
        d="M70 60 Q 10 60 -30 90 Q 25 80 70 60"
        fill="#000000"
        stroke="#000000"
        strokeWidth="2"
      />
      <circle cx="66" cy="65" r="7" fill="#FFFFFF" stroke="#000000" strokeWidth="2" />
      <circle cx="76" cy="63" r="6" fill="#9333EA" stroke="#000000" strokeWidth="2" />
    </svg>
  );
}

// ── Right Framing Palm Tree (Screenshot 1 & 13) ───────────────────────────────
export function GoaRightPalmTree() {
  return (
    <svg viewBox="0 0 160 380" fill="none" xmlns="http://www.w3.org/2000/svg" className="w-full h-auto">
      <g transform="scale(-1, 1) translate(-160, 0)">
        <path
          d="M-20 380 Q 20 220 50 140 Q 65 90 70 60"
          stroke="#000000"
          strokeWidth="18"
          strokeLinecap="round"
        />
        <path
          d="M-20 380 Q 20 220 50 140 Q 65 90 70 60"
          stroke="#000000"
          strokeWidth="10"
          strokeLinecap="round"
        />
        <path
          d="M70 60 Q 110 30 150 50 Q 120 75 70 60"
          fill="#000000"
          stroke="#000000"
          strokeWidth="2"
        />
        <path
          d="M70 60 Q 95 10 140 -5 Q 115 35 70 60"
          fill="#8FC93A"
          stroke="#000000"
          strokeWidth="2"
        />
        <path
          d="M70 60 Q 50 -10 60 -50 Q 40 -5 70 60"
          fill="#000000"
          stroke="#000000"
          strokeWidth="2"
        />
        <path
          d="M70 60 Q 20 15 -20 5 Q 15 45 70 60"
          fill="#8FC93A"
          stroke="#000000"
          strokeWidth="2"
        />
        <circle cx="66" cy="65" r="7" fill="#FFFFFF" stroke="#000000" strokeWidth="2" />
        <circle cx="76" cy="63" r="6" fill="#9333EA" stroke="#000000" strokeWidth="2" />
      </g>
    </svg>
  );
}

// ── Half Sun Artwork with Rays (Screenshot 1 & 2) ─────────────────────────────
export function GoaRisingSun() {
  return (
    <svg viewBox="0 0 300 150" fill="none" xmlns="http://www.w3.org/2000/svg" className="w-72 h-36">
      <line x1="150" y1="140" x2="150" y2="20" stroke="#FFFFFF" strokeWidth="3" strokeLinecap="round" opacity="0.6" />
      <line x1="150" y1="140" x2="70" y2="40" stroke="#FFFFFF" strokeWidth="3" strokeLinecap="round" opacity="0.6" />
      <line x1="150" y1="140" x2="230" y2="40" stroke="#FFFFFF" strokeWidth="3" strokeLinecap="round" opacity="0.6" />
      <line x1="150" y1="140" x2="20" y2="90" stroke="#FFFFFF" strokeWidth="3" strokeLinecap="round" opacity="0.6" />
      <line x1="150" y1="140" x2="280" y2="90" stroke="#FFFFFF" strokeWidth="3" strokeLinecap="round" opacity="0.6" />
      <circle cx="150" cy="150" r="110" fill="#000000" opacity="0.5" />
      <circle cx="150" cy="150" r="85" fill="#FFFFFF" />
      <circle cx="150" cy="150" r="65" fill="#9333EA" opacity="0.2" />
    </svg>
  );
}

// ── Goan Beach Scenery Illustration (Screenshot 2 & 6) ───────────────────────
export function GoaBeachScenery() {
  return (
    <div className="w-full relative overflow-hidden py-12 select-none">
      <svg
        viewBox="0 0 1200 240"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="w-full h-auto max-h-56 mx-auto"
      >
        <path d="M0 200 C300 180 600 220 1200 190 L1200 240 L0 240 Z" fill="#E8DFB8" />
        <path d="M0 215 C400 200 800 230 1200 210 L1200 240 L0 240 Z" fill="#FAF7DE" />
        <path d="M0 170 C400 150 800 185 1200 160 L1200 200 C600 220 300 180 0 200 Z" fill="#000000" opacity="0.4" />

        {/* Beach Shack */}
        <g transform="translate(480, 100)">
          <polygon points="30,0 210,0 240,40 0,40" fill="#8FC93A" stroke="#000000" strokeWidth="3" />
          <polygon points="40,-8 200,-8 215,0 25,0" fill="#FFFFFF" stroke="#000000" strokeWidth="2" />
          <rect x="25" y="40" width="190" height="70" fill="#FEFBEA" stroke="#000000" strokeWidth="3" />
          <line x1="45" y1="40" x2="45" y2="110" stroke="#000000" strokeWidth="3" />
          <line x1="195" y1="40" x2="195" y2="110" stroke="#000000" strokeWidth="3" />
          <line x1="120" y1="40" x2="120" y2="110" stroke="#000000" strokeWidth="3" />
          <rect x="60" y="12" width="120" height="20" rx="4" fill="#9333EA" stroke="#FEFBEA" strokeWidth="2" />
          <text x="120" y="26" textAnchor="middle" fill="#FEFBEA" fontSize="10" fontWeight="bold" fontFamily="monospace">
            MANGO SHACK
          </text>
        </g>

        {/* Beach Umbrellas */}
        <g transform="translate(240, 130)">
          <path d="M10 50 A 45 45 0 0 1 90 50 Z" fill="#9333EA" stroke="#000000" strokeWidth="2.5" />
          <path d="M35 50 A 45 45 0 0 1 65 50 Z" fill="#FFFFFF" />
          <line x1="50" y1="50" x2="50" y2="90" stroke="#000000" strokeWidth="3" />
        </g>

        <g transform="translate(860, 135)">
          <path d="M10 50 A 45 45 0 0 1 90 50 Z" fill="#FFFFFF" stroke="#000000" strokeWidth="2.5" />
          <path d="M35 50 A 45 45 0 0 1 65 50 Z" fill="#8FC93A" />
          <line x1="50" y1="50" x2="50" y2="85" stroke="#000000" strokeWidth="3" />
        </g>

        {/* Surfboards */}
        <g transform="translate(180, 150) rotate(-15)">
          <path d="M0 0 C10 -25 15 -25 25 0 L22 70 C12 75 8 75 0 70 Z" fill="#9333EA" stroke="#000000" strokeWidth="2" />
          <line x1="12" y1="-10" x2="12" y2="65" stroke="#FFFFFF" strokeWidth="2" />
        </g>
        <g transform="translate(200, 155) rotate(10)">
          <path d="M0 0 C10 -25 15 -25 25 0 L22 70 C12 75 8 75 0 70 Z" fill="#FFFFFF" stroke="#000000" strokeWidth="2" />
          <line x1="12" y1="-10" x2="12" y2="65" stroke="#8FC93A" strokeWidth="2" />
        </g>
      </svg>
    </div>
  );
}
