/**
 * NEXUS-II — Portfolio Overview (home page)
 * Server component — fetches data from Supabase at request time.
 *
 * Sections
 * ────────
 *  • 4 KPI cards: Total PnL | Today's PnL | Win Rate | Open Positions
 *  • Open positions table (live, realtime-refreshed via client component)
 *  • Recent closed trades table
 */
import { fetchPortfolioKPIs, fetchOpenTrades, fetchClosedTrades } from "@/lib/supabase";
import type { Trade } from "@/lib/supabase";
import { TrendingUp, TrendingDown, Activity, DollarSign } from "lucide-react";

// ── Helpers ───────────────────────────────────────────────────────────────

function fmtPnl(v: number): string {
  const sign = v >= 0 ? "+" : "";
  return `${sign}₹${Math.abs(v).toLocaleString("en-IN", { maximumFractionDigits: 0 })}`;
}

function fmtPct(v: number): string {
  return `${(v * 100).toFixed(1)}%`;
}

// ── KPI Card ──────────────────────────────────────────────────────────────

function KPICard({
  label,
  value,
  sub,
  positive,
  icon,
}: {
  label: string;
  value: string;
  sub?: string;
  positive?: boolean;
  icon: React.ReactNode;
}) {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 flex flex-col gap-3">
      <div className="flex items-center justify-between text-slate-400">
        <span className="text-xs font-semibold uppercase tracking-wider">{label}</span>
        <span className="opacity-60">{icon}</span>
      </div>
      <div
        className={`text-2xl font-bold ${
          positive === undefined
            ? "text-white"
            : positive
            ? "text-emerald-400"
            : "text-red-400"
        }`}
      >
        {value}
      </div>
      {sub && <div className="text-xs text-slate-500">{sub}</div>}
    </div>
  );
}

// ── Trade Table ───────────────────────────────────────────────────────────

function TradeRow({ trade }: { trade: Trade }) {
  const isBuy = trade.direction === "BUY";
  const pnl = trade.pnl ?? 0;
  return (
    <tr className="border-t border-slate-800 hover:bg-slate-800/40 transition-colors text-sm">
      <td className="py-3 px-4 font-semibold text-white">{trade.symbol}</td>
      <td className="py-3 px-4">
        <span
          className={`text-xs font-bold px-2 py-0.5 rounded ${
            isBuy ? "bg-emerald-900/60 text-emerald-400" : "bg-red-900/60 text-red-400"
          }`}
        >
          {trade.direction}
        </span>
      </td>
      <td className="py-3 px-4 text-slate-300">₹{trade.entry_price.toLocaleString("en-IN")}</td>
      <td className="py-3 px-4 text-slate-400">{trade.quantity}</td>
      <td className="py-3 px-4">
        {trade.conviction != null ? (
          <div className="flex items-center gap-2">
            <div className="h-1.5 w-24 bg-slate-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-indigo-500 rounded-full"
                style={{ width: `${(trade.conviction * 100).toFixed(0)}%` }}
              />
            </div>
            <span className="text-slate-400 text-xs">{(trade.conviction * 100).toFixed(0)}%</span>
          </div>
        ) : (
          <span className="text-slate-600">—</span>
        )}
      </td>
      <td className="py-3 px-4 text-slate-400 text-xs">{trade.agent_name}</td>
      <td
        className={`py-3 px-4 font-semibold ${
          pnl > 0 ? "text-emerald-400" : pnl < 0 ? "text-red-400" : "text-slate-400"
        }`}
      >
        {trade.status === "OPEN" ? "—" : fmtPnl(pnl)}
      </td>
      <td className="py-3 px-4 text-slate-500 text-xs">
        {new Date(trade.opened_at).toLocaleString("en-IN", {
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

// ── Page ──────────────────────────────────────────────────────────────────

export default async function PortfolioPage() {
  const [kpis, openTrades, closedTrades] = await Promise.all([
    fetchPortfolioKPIs(),
    fetchOpenTrades(),
    fetchClosedTrades(30),
  ]);

  const colHeaders = ["Symbol", "Dir", "Entry", "Qty", "Conviction", "Agent", "PnL", "Opened"];

  return (
    <div className="space-y-6">
      {/* Page title */}
      <div>
        <h1 className="text-xl font-bold text-white">Portfolio Overview</h1>
        <p className="text-slate-500 text-sm mt-0.5">
          Live positions and trade history — NSE/BSE
        </p>
      </div>

      {/* KPI row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <KPICard
          label="Total PnL"
          value={fmtPnl(kpis.total_pnl)}
          positive={kpis.total_pnl >= 0}
          icon={<DollarSign size={16} />}
        />
        <KPICard
          label="Today's PnL"
          value={fmtPnl(kpis.today_pnl)}
          positive={kpis.today_pnl >= 0}
          icon={<TrendingUp size={16} />}
        />
        <KPICard
          label="Win Rate"
          value={fmtPct(kpis.win_rate)}
          sub="closed trades"
          icon={<Activity size={16} />}
        />
        <KPICard
          label="Open Positions"
          value={String(kpis.open_positions)}
          icon={<TrendingDown size={16} />}
        />
      </div>

      {/* Open positions */}
      <section className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden">
        <div className="px-5 py-4 border-b border-slate-800 flex items-center justify-between">
          <h2 className="font-semibold text-white">Open Positions</h2>
          <span className="text-xs text-slate-500 bg-slate-800 px-2 py-0.5 rounded">
            {openTrades.length} active
          </span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="text-left text-xs text-slate-500 uppercase tracking-wider">
                {colHeaders.map((h) => (
                  <th key={h} className="py-2 px-4 font-semibold">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {openTrades.length === 0 ? (
                <tr>
                  <td colSpan={8} className="py-10 text-center text-slate-600 text-sm">
                    No open positions
                  </td>
                </tr>
              ) : (
                openTrades.map((t) => <TradeRow key={t.id} trade={t} />)
              )}
            </tbody>
          </table>
        </div>
      </section>

      {/* Recent closed trades */}
      <section className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden">
        <div className="px-5 py-4 border-b border-slate-800">
          <h2 className="font-semibold text-white">Recent Closed Trades</h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="text-left text-xs text-slate-500 uppercase tracking-wider">
                {colHeaders.map((h) => (
                  <th key={h} className="py-2 px-4 font-semibold">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {closedTrades.length === 0 ? (
                <tr>
                  <td colSpan={8} className="py-10 text-center text-slate-600 text-sm">
                    No closed trades yet
                  </td>
                </tr>
              ) : (
                closedTrades.map((t) => <TradeRow key={t.id} trade={t} />)
              )}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
