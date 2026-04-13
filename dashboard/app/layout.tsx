import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "NEXUS-II Dashboard",
  description: "NEXUS-II v2.1 — AI Trading Bot for NSE/BSE",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-slate-950 text-slate-100 min-h-screen antialiased">
        {/* Top nav */}
        <nav className="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-50">
          <div className="max-w-7xl mx-auto px-4 h-14 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="text-indigo-400 font-bold text-lg tracking-tight">
                NEXUS-II
              </span>
              <span className="text-slate-500 text-sm">v2.1</span>
            </div>
            <div className="flex items-center gap-6 text-sm font-medium">
              <a href="/" className="text-slate-300 hover:text-white transition-colors">
                Portfolio
              </a>
              <a href="/agents" className="text-slate-300 hover:text-white transition-colors">
                Agents
              </a>
              <a href="/signals" className="text-slate-300 hover:text-white transition-colors">
                Signals
              </a>
              <a href="/calibration" className="text-slate-300 hover:text-white transition-colors">
                Calibration
              </a>
            </div>
          </div>
        </nav>
        <main className="max-w-7xl mx-auto px-4 py-6">{children}</main>
      </body>
    </html>
  );
}
