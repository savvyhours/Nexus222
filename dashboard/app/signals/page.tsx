/**
 * NEXUS-II — Signal History
 * Shows the signal_log table: per-symbol DynamicSignalScorer results
 * with component breakdown (technical, sentiment, fundamental, macro,
 * candlestick, ml_qlib, debate_conviction).
 */
import { fetchRecentSignals } from "@/lib/supabase";
import type { SignalLog } from "@/lib/supabase";

// ── Constants ─────────────────────────────────────────────────────────────

const COMPONENT_KEYS = [
  "technical",
  "sentiment",
  "fundamental",
  "macro",
  "candlestick",
  "ml_qlib",
  "debate_conviction",
] as const;

const REGIME_COLORS: Record<string, string> = {
  TRENDING:       "text-blue-400 bg-blue-900/40",
  MEAN_REVERTING: "text-violet-400 bg-violet-900/40",
  HIGH_VOL:       "text-orange-400 bg-orange-900/40",
  LOW_VOL:        "text-teal-400 bg-teal-900/40",
  CRISIS:         "text-red-400 bg-red-900/40",
};

// ── Helpers ───────────────────────────="──────────────────────────────────

function scoreColor(score: number | null): string {
  if (score == null) return "text-slate-500";
  if (score >= 0.5) return "text-emerald-400";
  if (score > 0) return "text-emerald-600";
  if (score <= -0.5) return "text-red-400";
  if (score < 0) return "text-red-600";
  return "text-slate-400";
}

function fmt(v: number | null, decimals = 2): string {
  if (v == null) return "—";
  const s = v >= 0 ? "+" : "";
  return `${s}${v.toFixed(decimals)}`;
}

// ── Component spark bar ───────────────────────────────────────────────────

function ComponentBar({ value }: { value: number }) {
  const pct = Math.abs(value) * 100;
  const positive = value >= 0;
  return (
    <div className="flex items-center gap-1 w-20">
      <div className="flex-1 h-1.5 bg-slate-800 rounded-full overflow-hidden flex">
        {positive ? (
          <>
            <div className="w-1/2" />
            <div
              className="bg-emerald-500 rounded-full"
              style={{ width: `${pct / 2}%` }}
            />
          </>
        ) : (
          <>
            <div
              className="bg-red-500 rounded-full ml-auto"
              style={{ width: `${pct / 2}%` }}
            />
            <div className="w-1/2" />
          </>
        )}
      </div>
      <span className={`text-xs font-mono w-10 text-right ${positive ? "text-emerald-400" : "text-red-400"}`}>
        {fmt(value, 2)}
      </span>
    </div>
  );
}

// ── Signal row ────────────────────────────────────────────────────────────

function SignalRow({ signal }: { signal: SignalLog }) {
  const score = signal.signal_score ?? 0;
  const triggered = signal.triggered ?? false;
  const regime = signal.regime ?? "UNKNOWN";
  const regimeClass = REGIME_COLORS[regime] ?? "text-slate-400 bg-slate-800/40";
  const components = signal.components ?? {};

  return (
    <tr className="border-t border-slate-800 hover:bg-slate-800/30 transition-colors text-xs">
      {/* Symbol */}
      <td className="py-3 px-4 font-semibold text-white text-sm">{signal.symbol}</td>

      {/* Score */}
      <td className="py-3 px-4">
        <span className={`font-mono font-bold text-sm ${scoreColor(signal.signal_score)}`}>
          {fmt(signal.signal_score)}
        </span>
      </td>

      {/* Triggered */}
      <td className="py-3 px-4">
        {triggered ? (
          <span className="text-xs px-2 py-0.5 rounded bg-indigo-900/60 text-indigo-300 font-semibold">
            TRIGGERED
          </span>
        ) : (
          <span className="text-slate-600 text-xs">—</span>
        )}
      </td>

      {/* Regime */}
      <td className="py-3 px-4">
        <span className={`text-xs px-2 py-0.5 rounded font-semibold ${regimeClass}`}>
          {regime}
        </span>
      </td>

      {/* Threshold */}
      <td className="py-3 px-4 text-slate-400 font-mono">
        {signal.threshold != null ? signal.threshold.toFixed(2) : "—"}
      </td>

      {/* Component breakdown — 7 mini bars */}
      {COMPONENT_KEYS.map((k) => (
        <td key={k} className="py-3 px-2">
          {components[k] != null ? (
            <ComponentBar value={components[k]} />
          ) : (
            <span className="text-slate-700">—</span>
          )}
        </td>
      ))}

      {/* Timestamp */}
      <td className="py-3 px-4 text-slate-600 text-xs whitespace-nowrap">
        {new Date(signal.created_at).toLocaleString("en-IN", {
          month: "short",
          day: "numeric",
          hour: "2-digit",
          minute: "2-digit",
          timeZone: "Asia/Kolkata",
        })}
      </td>
    </tr>
  );
}

// ── Stats summary ─────────────────────────────────────────────────────────

function SignalStats({ signals }: { signals: SignalLog[] }) {
  const triggered = signals.filter((s) => s.triggered).length;
  const buys = signals.filter((s) => (s.signal_score ?? 0) > 0 && s.triggered).length;
  const sells = signals.filter((s) => (s.signal_score ?? 0) < 0 && s.triggered).length;
  const regimes = signals.reduce<Record<string, number>>((acc, s) => {
    const r = s.regime ?? "UNKNOWN";
    acc[r] = (acc[r] ?? 0) + 1;
    return acc;
  }, {});

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      {[
        { label: "Total Signals", value: signals.length },
        { label: "Triggered", value: triggered },
        { label: "BUY signals", value: buys },
        { label: "SELL signals", value: sells },
      ].map(({ label, value }) => (
        <div key={label} className="bg-slate-900 border border-slate-800 rounded-xl p-4">
          <div className="text-xs text-slate-500 uppercase tracking-wider mb-2">{label}</div>
          <div className="text-2xl font-bold text-white">{value}</div>
        </div>
      ))}
    </div>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────

export default async function SignalsPage() {
  const signals = await fetchRecentSignals(100);

  const colHeaders = [
    "Symbol",
    "Score",
    "Status",
    "Regime",
    "Threshold",
    ...COMPONENT_KEYS.map((k) =>
      k === "debate_conviction" ? "Debate" : k.charAt(0).toUpperCase() + k.slice(1)
    ),
    "Time (IST)",
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-xl font-bold text-white">Signal History</h1>
        <p className="text-slate-500 text-sm mt-0.5">
          DynamicSignalScorer output — 7-component breakdown — last 100 signals
        </p>
      </div>

      {/* Stats */}
      <SignalStats signals={signals} />

      {/* Table */}
      <section className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden">
        <div className="px-5 py-4 border-b border-slate-800">
          <h2 className="font-semibold text-white">Component Breakdown</h2>
          <p className="text-slate-600 text-xs mt-1">
            Each bar shows the raw component value in [−1, +1]. Green = bullish, Red = bearish.
          </p>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="text-left text-xs text-slate-500 uppercase tracking-wider">
                {colHeaders.map((h) => (
                  <th key={h} className="py-2 px-4 font-semibold whitespace-nowrap">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {signals.length === 0 ? (
                <tr>
                  <td
                    colSpan={colHeaders.length}
                    className="py-12 text-center text-slate-600 text-sm"
                  >
                    No signals yet — the signal engine will populate this table once the bot runs
                  </td>
                </tr>
              ) : (
                signals.map((s) => <SignalRow key={s.id} signal={s} />)
              )}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
