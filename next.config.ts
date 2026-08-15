import type { NextConfig } from "next";

const BACKEND_URL =
  process.env.NEXT_PUBLIC_API_BASE ||
  (process.env.NODE_ENV === "production" ? "" : "http://127.0.0.1:8000");

// Only add the proxy rewrite when we have a valid absolute URL.
const isValidUrl =
  BACKEND_URL.startsWith("http://") || BACKEND_URL.startsWith("https://");

const nextConfig: NextConfig = {
  async rewrites() {
    if (!isValidUrl) return [];
    return [
      {
        source: "/api/:path*",
        destination: `${BACKEND_URL}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;

