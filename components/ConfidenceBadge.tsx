"use client";

interface ConfidenceBadgeProps {
  level: "high" | "medium" | "refused";
  score?: number;
  showScore?: boolean;
}

const CONFIG = {
  high: {
    label: "HIGH",
    className: "badge-high",
    dot: "bg-forest",
    description: "Strong evidence + grounding passed",
  },
  medium: {
    label: "MEDIUM",
    className: "badge-medium",
    dot: "bg-sunshine",
    description: "Borderline evidence or grounding",
  },
  refused: {
    label: "REFUSED",
    className: "badge-refused",
    dot: "bg-hibiscus",
    description: "Insufficient evidence",
  },
};

export default function ConfidenceBadge({ level, score, showScore }: ConfidenceBadgeProps) {
  const cfg = CONFIG[level] || CONFIG.refused;

  return (
    <div className="flex items-center gap-2" title={cfg.description}>
      <div className={`flex items-center gap-1.5 ${cfg.className} font-mono`}>
        <span className={`w-1.5 h-1.5 rounded-full ${cfg.dot}`} />
        {cfg.label}
      </div>
      {showScore && score !== undefined && (
        <span className="text-muted-text text-xs font-mono">
          ({(score * 100).toFixed(0)}%)
        </span>
      )}
    </div>
  );
}
