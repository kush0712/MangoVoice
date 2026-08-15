"use client";

interface TranscriptCardProps {
  text: string;
  language?: string | null;
  visible: boolean;
}

const LANG_LABELS: Record<string, string> = {
  "hi": "हिन्दी",
  "hi-IN": "हिन्दी",
  "en": "English",
  "en-IN": "English",
  "codemix": "Hinglish",
  "auto": "Auto",
};

export default function TranscriptCard({ text, language, visible }: TranscriptCardProps) {
  if (!visible || !text) return null;

  const langLabel = language ? (LANG_LABELS[language] || language.toUpperCase()) : null;

  return (
    <div className="card-glass rounded-card p-5 animate-slide-up">
      <div className="flex items-center justify-between mb-3">
        <div className="eyebrow text-cream/60">You asked</div>
        {langLabel && (
          <div className="pill-outline text-cream/60 border-cream/20 text-xs">
            {langLabel}
          </div>
        )}
      </div>
      <blockquote className="text-cream text-base leading-relaxed font-mono">
        &ldquo;{text}&rdquo;
      </blockquote>
    </div>
  );
}
