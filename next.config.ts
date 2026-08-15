import type { NextConfig } from "next";

const BACKEND_URL = process.env.NEXT_PUBLIC_API_BASE ?? "";

// Only add the proxy rewrite when we have a valid absolute URL.
// This prevents the Vercel build from failing when the env var is missing.
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
