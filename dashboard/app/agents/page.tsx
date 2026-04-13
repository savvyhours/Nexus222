/**
 * NEXUS-II — Agent Leaderboard
 * Shows real-time Sharpe-weighted rankings, win rates, and dynamic weights
 * for all 10 strategy agents.
 *
 * Sorted by: current weight (proxy for calibrated performance) desc.
 */
import { fetchLatestAgentPerformance } from "@/lib/supabase";
import type { AgentPerformance } from "@/lib/supabase";

// ── Agent metadata (display names + descriptions) ─────────────────────────

const AGENT_META: Record<string, { label: string; description: string; color: string }> = {
  scalper:        { label: "Scalper",         description: "VWAP + EMA(9/21) intraday",       color: "bg-cyan-500" },
  trend:          { label: "Trend",           description: "EMA ribbon + ADX + MACD",          color: "bg-blue-500" },
  mean_reversion: { label: "Mean Reversion",  description: "Bollinger + Z-score + RSI div",    color: "bg-violet-500" },
  sentiment:      { label: "Sentiment",       description: "NLP news/social score",            color: "bg-pink-500" },
  fundamentals:   { label: "Fundamentals",    description: "P/E + ROE + EPS growth",           color: "bg-amber-500" },
  macro:          { label: "Macro",           description: "FII flows + VIX + USD/INR",        color: "bg-orange-500" },
  options:        { label: "Options",         description: "IV crush + theta + max pain",      color: "bg-red-500" },
  pattern:        { label: "Pattern",         description: "Candlestick composites",           color: "bg-emerald-500" },
  quant:          { label: "Quant",           description: "Qlib ML rank percentile",          color: "bg-teal-500" },
  etf:            { label: "ETF",             description: "NAV arb + sector rotation",        color: "bg-lime-500" },
};

// ── Helpers ───────────────────────────────────────────────────────────────

function pct(v: number | null): string {
  if (v == null) return "—";
  return `${(v * 100).toFixed(1)}%`;
}

function num(v: number | null, decimals = 2): string {
  if (v == null) return "—";
  return v.toFixed(decimals);
}

function pnlStr(v: number | null): string {
  if (v == null) return "—";
  const sign = v >= 0 ? "+" : "";
  return `${sign}₹${Math.abs(v).toLocaleString("en-IN", { maximumFractionDigits: 0 })}`;
}

// ── Agent row ─────────────────────────────────────────────────────────────

function AgentRow({
  agent,
  rank,
}: {
  agent: AgentPerformance;
  rank: number;
}) {
  const meta = AGENT_META[agent.agent_name] ?? {
    label: agent.agent_name,
    description: "",
    color: "bg-slate-500",
  };
  const weight = agent.weight ?? 0;
  const winRate = agent.win_rate ?? 0;
  const pnl = agent.pnl_total ?? 0;

  return (
    <tr className="border-t border-slate-800 hover:bg-slate-800/40 transition-colors">
      {/* Rank */}
      <td className="py-4 px-4 text-slate-500 font-mono text-sm">{rank}</td>

      {/* Agent */}
      <td className="py-4 px-4">
        <div className="flex items-center gap-3">
          <span className={`w-2 h-2 rounded-full ${meta.color}`} />
          <div>
            <div className="font-semibold text-white text-sm">{meta.label}</div>
            <div className="text-slate-500 text-xs">{meta.description}</div>
          </div>
        </div>
      </td>

      {/* Dynamic weight bar */}
      <td className="py-4 px-4">
        <div className="flex items-center gap-2">
          <div className="h-2 w-28 bg-slate-800 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full ${meta.color}`}
              style={{ width: `${(weight * 100).toFixed(0)}%` }}
            />
          </div>
          <span className="text-slate-300 text-xs font-mono">{pct(weight)}</span>
        </div>
      </td>

      {/* Sharpe */}
      <td className="py-4 px-4">
        <span
          className={`font-mono text-sm ${
            (agent.sharpe_30d ?? 0) >= 1
              ? "text-emerald-400"
              : (agent.sharpe_30d ?? 0) >= 0
              ? "text-slate-300"
              : "text-red-400"
          }`}
        >
          {num(agent.sharpe_30d)}
        </span>
      </td>

      {/* Win rate */}
      <td className="py-4 px-4">
        <span
          className={`text-sm font-mono ${
            winRate >= 0.55 ? "text-emerald-400" : winRate >= 0.45 ? "text-slate-300" : "text-red-400"
          }`}
        >
          {pct(agent.win_rate)}
        </span>
      </td>

      {/* Total PnL */}
      <td className="py-4 px-4">
        <span
          className={`text-sm font-semibold ${
            pnl >= 0 ? "text-emerald-400" : "text-red-400"
          }`}
        >
          {pnlStr(agent.pnl_total)}
        </span>
      </td>

      {/* Trades */}
      <td className="py-4 px-4 text-slate-400 text-sm font-mono">
        {agent.trades_count ?? "—"}
      </td>

      {/* Date */}
      <td className="py-4 px-4 text-slate-600 text-xs">{agent.date}</td>
    </tr>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────

export default async function AgentsPage() {
  const agents = await fetchLatestAgentPerformance();

  // Sort by calibrated weight desc, then Sharpe desc
  const sorted = [...agents].sort((a, b) => {
    const wDiff = (b.weight ?? 0) - (a.weight ?? 0);
    if (Math.abs(wDiff) > 0.001) return wDiff;
    return (b.sharpe_30d ?? 0) - (a.sharpe_30d ?? 0);
  });

  // Aggregate KPIs
  const totalWeight = sorted.reduce((s, a) => s + (a.weight ?? 0), 0);
  const topAgent = sorted[0];
  const avgSharpe =
    sorted.length
      ? sorted.reduce((s, a) => s + (a.sharpe_30d ?? 0), 0) / sorted.length
      : 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-xl font-bold text-white">Agent Leaderboard</h1>
        <p className="text-slate-500 text-sm mt-0.5">
          Dynamic weights set by WeightCalibrationAgent · 15-min TTL cache
        </p>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
          <div className="text-xs text-slate-500 uppercase tracking-wider mb-2">Active Agents</div>
          <div className="text-2xl font-bold text-white">{sorted.length}</div>
        </div>
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
          <div className="text-xs text-slate-500 uppercase tracking-wider mb-2">Top Agent</div>
          <div className="text-lg font-bold text-indigo-400">
            {topAgent ? (AGENT_META[topAgent.agent_name]?.label ?? topAgent.agent_name) : "—"}
          </div>
          {topAgent?.weight != null && (
            <div className="text-xs text-slate-500 mt-1">
              weight: {pct(topAgent.weight)}
            </div>
          )}
        </div>
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
          <div className="text-xs text-slate-500 uppercase tracking-wider mb-2">Avg Sharpe (30d)</div>
          <div
            className={`text-2xl font-bold ${avgSharpe >= 1 ? "text-emerald-400" : "text-slate-300"}`}
          >
            {avgSharpe.toFixed(2)}
          </div>
        </div>
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
          <div className="text-xs text-slate-500 uppercase tracking-wider mb-2">Total Weight Σ</div>
          <div className="text-2xl font-bold text-white">{pct(totalWeight)}</div>
        </div>
      </div>

      {/* Leaderboard table */}
      <section className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden">
        <div className="px-5 py-4 border-b border-slate-800">
          <h2 className="font-semibold text-white">Rankings by Calibrated Weight</h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="text-left text-xs text-slate-500 uppercase tracking-wider">
                {["#", "Agent", "Weight", "Sharpe 30d", "Win Rate", "Total PnL", "Trades", "As of"].map(
                  (h) => (
                    <th key={h} className="py-2 px-4 font-semibold">
                      {h}
                    </th>
                  )
                )}
              </tr>
            </thead>
            <tbody>
              {sorted.length === 0 ? (
                <tr>
                  <td colSpan={8} className="py-12 text-center text-slate-600 text-sm">
                    No performance data yet — agents will populate this table after their first trading day
                  </td>
                </tr>
              ) : (
                sorted.map((a, i) => <AgentRow key={a.id} agent={a} rank={i + 1} />)
              )}
            </tbody>
          </table>
        </div>
      </section>

      {/* Weight distribution visual */}
      {sorted.length > 0 && (
        <section className="bg-slate-900 border border-slate-800 rounded-xl p-5">
          <h2 className="font-semibold text-white mb-4">Weight Distribution</h2>
          <div className="space-y-2">
            {sorted.map((a) => {
              const meta = AGENT_META[a.agent_name] ?? { label: a.agent_name, color: "bg-slate-500" };
              const w = a.weight ?? 0;
              const barPct = totalWeight > 0 ? (w / totalWeight) * 100 : 0;
              return (
                <div key={a.id} className="flex items-center gap-3">
                  <div className="w-28 text-xs text-slate-400 text-right shrink-0">
                    {meta.label}
                  </div>
                  <div className="flex-1 h-5 bg-slate-800 rounded overflow-hidden">
                    <div
                      className={`h-full ${meta.color} opacity-80 transition-all`}
                      style={{ width: `${barPct.toFixed(1)}%` }}
                    />
                  </div>
                  <div className="w-12 text-xs text-slate-400 font-mono">{pct(w)}</div>
                </div>
              );
            })}
          </div>
        </section>
      )}
    </div>
  );
}
