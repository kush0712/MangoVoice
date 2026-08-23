"use client";

import React, { useState } from "react";
import { Plus, Minus } from "lucide-react";
import { GoaFloralDivider, GoaBeachScenery, GoaTextileRibbon } from "./GoaIllustrations";

const FAQS_DATA = [
  {
    q: "How does MangoVoice achieve sub-200ms RAG core latency?",
    a: "We avoid multi-second Python framework overhead by running a zero-dependency async pipeline. Query embeddings are generated locally via FastEmbed ONNX (P50 38.99ms), hybrid retrieval runs in-process via zero-copy LanceDB + flat inverted BM25 (P50 9.39ms), and extractive fallback answers in 0.25ms with instant grounding verifier (0.13ms), delivering a consistent 48.97ms P50 RAG core latency (P99 62.59ms, full N=500 sweep P50 53.58ms).",
  },
  {
    q: "How does MangoVoice handle Hindi, English, and Hinglish codemix?",
    a: "Speech-to-text uses Sarvam Saaras v3 configured with Hindi-English codemix mode. Dense retrieval uses paraphrase-multilingual-MiniLM-L12-v2 (384-dimensional multilingual embeddings) that projects Hindi and English concepts into the same vector space, blended with BM25 via Reciprocal Rank Fusion (RRF).",
  },
  {
    q: "Why evaluate 5 different chunking strategies?",
    a: "MSMARCO-XI contains heterogeneous passage lengths. By benchmarking Canonical, Sentence Windows, Fixed Token, Semantic Boundary Splitting, and Parent-Child Hierarchical on the official validation split, we empirically proved that Parent-Child maximizes both retrieval accuracy (R@10: 0.93) and LLM context coherence.",
  },
  {
    q: "How does the system prevent hallucinations and prompt injection?",
    a: "We implement 4 distinct guardrail layers: Layer 1 (deterministic regex for jailbreak patterns), Layer 2 (Llama Prompt Guard 2 22M safety classifier), Layer 3 (calibrated confidence gate that refuses low-evidence queries), and Layer 4 (post-generation sentence-level cosine similarity + entity overlap verifier).",
  },
  {
    q: "What happens if the question is off-topic or unanswerable?",
    a: "Refusal is a first-class citizen in MangoVoice. If the confidence gate or Groq tool-contract detects insufficient evidence, the system refuses gracefully with an informative explanation instead of hallucinating.",
  },
];

export default function GoaFaqAndFooter() {
  const [openFaq, setOpenFaq] = useState<number | null>(0);

  return (
    <div className="relative z-10 pt-10">
      {/* ── FAQ Section ── */}
      <section id="faqs-section" className="max-w-5xl mx-auto px-4 py-12">
        <div className="text-center mb-6">
          <h3 className="text-4xl md:text-6xl font-hhg-title text-cream shadow-hhg-text uppercase tracking-tight">
            FAQs
          </h3>
        </div>

        {/* Goan Floral Motifs Divider */}
        <GoaFloralDivider />

        {/* Accordion Questions */}
        <div className="space-y-4 max-w-4xl mx-auto">
          {FAQS_DATA.map((faq, idx) => {
            const isOpen = openFaq === idx;
            return (
              <div key={idx} className="border-b border-cream/15 pb-4">
                <button
                  onClick={() => setOpenFaq(isOpen ? null : idx)}
                  className="w-full flex items-center justify-between py-3 text-left group transition-all cursor-pointer"
                >
                  <span className="font-hhg-title text-lg md:text-2xl text-cream group-hover:text-sunshine transition-colors leading-snug">
                    {faq.q}
                  </span>
                  <span className="w-8 h-8 rounded-full border border-cream/50 flex items-center justify-center flex-shrink-0 ml-4 group-hover:border-sunshine group-hover:text-sunshine text-cream transition-all">
                    {isOpen ? <Minus size={18} /> : <Plus size={18} />}
                  </span>
                </button>

                {isOpen && (
                  <div className="pt-2 pb-4 pr-8 animate-fade-in">
                    <p className="font-hhg-mono text-xs md:text-sm text-cream/80 leading-relaxed">
                      {faq.a}
                    </p>
                  </div>
                )}

                {/* Floral separator */}
                {idx < FAQS_DATA.length - 1 && <GoaFloralDivider className="my-3 opacity-30" />}
              </div>
            );
          })}
        </div>
      </section>

      {/* ── Less Noise. More Signal Callout ── */}
      <section className="max-w-5xl mx-auto px-4 py-16">
        <div className="grid grid-cols-1 md:grid-cols-12 gap-8 items-center border-t border-b border-cream/20 py-12">
          <div className="md:col-span-5">
            <h3 className="text-4xl md:text-6xl font-hhg-title text-cream leading-none tracking-tight">
              Less Noise.
              <br />
              <span className="text-sunshine">More Signal.</span>
            </h3>
          </div>

          <div className="md:col-span-7 space-y-6">
            <p className="font-hhg-mono text-xs md:text-sm text-sunshine leading-relaxed">
              Most voice bots suffer from 4-second latency and ungrounded hallucinations.
              MangoVoice provides sub-200ms voice RAG with empirical retrieval evaluation,
              Sarvam Indic speech processing, and strict factual tool-contract generation.
            </p>

            <div className="flex flex-wrap items-center gap-4">
              <a
                href="#studio-section"
                className="btn-hhg-textile"
              >
                TRY MANGOVOICE
              </a>
              <span className="text-xs font-hhg-mono text-hibiscus font-bold uppercase tracking-wider">
                #RAGInGoa · MULTILINGUAL INDIC RAG
              </span>
            </div>
          </div>
        </div>
      </section>

      {/* ── Beach Scenery Artwork ── */}
      <div className="mt-8">
        <GoaBeachScenery />
      </div>

      {/* ── Goan Textile Ribbon Band ── */}
      <GoaTextileRibbon />

      {/* ── Footer ── */}
      <footer className="bg-forest-dark py-10 px-6 border-t border-cream/10">
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row items-center justify-between gap-6 text-center md:text-left">
          {/* Brand & Stack */}
          <div>
            <div className="flex items-center justify-center md:justify-start gap-2.5 mb-2">
              <img
                src="/mango-emblem.png"
                alt="MangoVoice"
                className="w-8 h-8 object-contain drop-shadow-sm"
              />
              <span className="font-hhg-title text-2xl text-cream font-bold">MangoVoice</span>
              <span className="text-xs font-hhg-mono bg-hibiscus text-cream px-2.5 py-0.5 rounded-full font-bold uppercase">
                #RAGInGoa
              </span>
            </div>
            <p className="text-xs font-hhg-mono text-cream/60">
              Voice-Enabled Indic RAG · Hindi · English · Hinglish · MSMARCO-XI
            </p>
          </div>

          {/* Tech Stack Pills */}
          <div className="flex flex-wrap items-center justify-center md:justify-end gap-2 text-xs font-hhg-mono text-sunshine">
            <span className="px-2.5 py-1 rounded-full border border-cream/20 bg-forest/50">Sarvam Saaras v3</span>
            <span className="px-2.5 py-1 rounded-full border border-cream/20 bg-forest/50">LanceDB</span>
            <span className="px-2.5 py-1 rounded-full border border-cream/20 bg-forest/50">FastEmbed</span>
            <span className="px-2.5 py-1 rounded-full border border-cream/20 bg-forest/50">Groq GPT-OSS 20B</span>
          </div>
        </div>

        <div className="max-w-6xl mx-auto mt-8 pt-6 border-t border-cream/10 flex flex-col sm:flex-row items-center justify-between text-[11px] font-hhg-mono text-cream/50 gap-2">
          <div>© {new Date().getFullYear()} MANGOVOICE · OPEN SOURCE VOICE RAG STUDIO.</div>
          <div>POWERED BY SARVAM AI · LANCE DB · FAST EMBED · GROQ</div>
        </div>
      </footer>
    </div>
  );
}
