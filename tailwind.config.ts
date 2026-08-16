import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./lib/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        forest: {
          DEFAULT: "#000000",
          deep: "#000000",
        },
        sunshine: "#FFFF00",
        hibiscus: "#FF0080",
        cream: "#00FFFF",
        ink: "#1A1A17",
        muted: {
          fill: "#D8D5C7",
          text: "#827F77",
        },
      },
      fontFamily: {
        display: ["Playfair Display", "Georgia", "serif"],
        mono: ["Space Mono", "JetBrains Mono", "ui-monospace", "monospace"],
      },
      borderRadius: {
        card: "24px",
        tag: "10px",
        pill: "999px",
      },
      boxShadow: {
        card: "6px 8px 0px rgba(0,0,0,0.15)",
        "card-hover": "8px 10px 0px rgba(0,0,0,0.20)",
        "inner-glow": "inset 0 0 30px rgba(255, 255, 0,0.1)",
      },
      animation: {
        "pulse-slow": "pulse 3s ease-in-out infinite",
        "bounce-slow": "bounce 2s infinite",
        "spin-slow": "spin 3s linear infinite",
        "waveform": "waveform 1.2s ease-in-out infinite",
        "fade-in": "fadeIn 0.4s ease-out forwards",
        "slide-up": "slideUp 0.5s ease-out forwards",
        "ping-slow": "ping 2s cubic-bezier(0, 0, 0.2, 1) infinite",
        "count-up": "countUp 2s ease-out forwards",
      },
      keyframes: {
        waveform: {
          "0%, 100%": { transform: "scaleY(0.3)" },
          "50%": { transform: "scaleY(1)" },
        },
        fadeIn: {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
        slideUp: {
          "0%": { opacity: "0", transform: "translateY(20px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        countUp: {
          "0%": { opacity: "0", transform: "translateY(10px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
    },
  },
  plugins: [],
};

export default config;
