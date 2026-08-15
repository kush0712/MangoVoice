"use client";

import { Zap } from "lucide-react";

interface DemoQuery {
  id: string;
  label: string;
  text: string;
  type: "answerable" | "off-topic" | "unsafe" | "ambiguous";
}

const DEMO_QUERIES: DemoQuery[] = [
  {
    id: "d1",
    label: "Manhattan Project",
    text: "What is the immediate impact of the success of the Manhattan Project?",
    type: "answerable",
  },
  {
    id: "d2",
    label: "Indian Independence",
    text: "भारत को आजादी कब मिली और किसने इसमें मुख्य भूमिका निभाई?",
    type: "answerable",
  },
  {
    id: "d3",
    label: "Hinglish query",
    text: "Can you tell me about World War II ka impact on European countries?",
    type: "answerable",
  },
  {
    id: "d4",
    label: "Off-topic (should refuse)",
    text: "What is the best pizza topping combination?",
    type: "off-topic",
  },
  {
    id: "d5",
    label: "Ambiguous query",
    text: "Tell me about the major event in 1945",
    type: "ambiguous",
  },
];

const TYPE_STYLE: Record<string, string> = {
  answerable: "border-sunshine/30 hover:border-sunshine/60 hover:bg-sunshine/5",
  "off-topic": "border-hibiscus/30 hover:border-hibiscus/60 hover:bg-hibiscus/5",
  unsafe: "border-hibiscus/30 hover:border-hibiscus/60 hover:bg-hibiscus/5",
  ambiguous: "border-cream/20 hover:border-cream/40 hover:bg-cream/5",
};

const TYPE_LABEL: Record<string, string> = {
  answerable: "answerable",
  "off-topic": "off-topic",
  unsafe: "guardrail",
  ambiguous: "ambiguous",
};

interface DemoQueriesProps {
  onSelect: (text: string) => void;
  disabled?: boolean;
}

export default function DemoQueries({ onSelect, disabled }: DemoQueriesProps) {
  return (
    <div>
      <div className="flex items-center gap-2 mb-3">
        <Zap size={12} className="text-sunshine" />
        <span className="eyebrow text-cream/50 text-xs">Try a demo query</span>
      </div>
      <div className="flex flex-wrap gap-2">
        {DEMO_QUERIES.map((q) => (
          <button
            key={q.id}
            id={`demo-query-${q.id}`}
            onClick={() => !disabled && onSelect(q.text)}
            disabled={disabled}
            title={q.text}
            className={`
              text-xs font-mono text-cream/60 border rounded-tag px-3 py-1.5
              transition-all duration-200 text-left max-w-[180px] truncate
              disabled:opacity-40 disabled:cursor-not-allowed
              ${TYPE_STYLE[q.type]}
            `}
          >
            <span className="text-muted-text text-[10px] block uppercase tracking-wider mb-0.5">
              {TYPE_LABEL[q.type]}
            </span>
            {q.label}
          </button>
        ))}
      </div>
    </div>
  );
}
