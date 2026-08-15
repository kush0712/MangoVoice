import type { Metadata } from "next";
import { Abril_Fatface, Bodoni_Moda, Space_Mono, Yatra_One } from "next/font/google";
import "./globals.css";

const abril = Abril_Fatface({
  subsets: ["latin"],
  weight: "400",
  variable: "--font-fatface",
  display: "swap",
});

const bodoni = Bodoni_Moda({
  subsets: ["latin"],
  weight: ["400", "700", "800", "900"],
  style: ["normal", "italic"],
  variable: "--font-display",
  display: "swap",
});

const spaceMono = Space_Mono({
  subsets: ["latin"],
  weight: ["400", "700"],
  variable: "--font-mono",
  display: "swap",
});

const yatraOne = Yatra_One({
  subsets: ["devanagari", "latin"],
  weight: "400",
  variable: "--font-devanagari",
  display: "swap",
});

export const metadata: Metadata = {
  title: "MangoVoice 🥭 — Voice-Enabled Indic RAG",
  description:
    "Speak a question. Get a grounded answer. Voice-enabled RAG over MSMARCO-XI in Hindi, English, and Hinglish. Powered by Sarvam Saaras v3, LanceDB, and Groq.",
  keywords: ["RAG", "voice", "Hindi", "Indic", "retrieval", "AI", "MangoVoice"],
  openGraph: {
    title: "MangoVoice — Voice-Enabled Indic RAG",
    description: "Speak a question. Get a grounded answer you can verify.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html
      lang="en"
      className={`${abril.variable} ${bodoni.variable} ${spaceMono.variable} ${yatraOne.variable}`}
    >
      <body className="font-mono antialiased selection:bg-sunshine selection:text-forest">
        {children}
      </body>
    </html>
  );
}
