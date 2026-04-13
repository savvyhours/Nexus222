/**
 * NEXUS-II v2.1 — Supabase client helpers
 *
 * Usage (server component / route handler):
 *   import { supabase } from "@/lib/supabase"
 *
 * All table types match the schema in supabase/migrations/001_initial_schema.sql
 */
import { createClient } from "@supabase/supabase-js";

// ── env vars (set in Vercel project settings) ─────────────────────────────
const SUPABASE_URL = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const SUPABASE_ANON_KEY = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;

if (!SUPABASE_URL || !SUPABASE_ANON_KEY) {
  throw new Error(
    "Missing NEXT_PUBLIC_SUPABASE_URL or NEXT_PUBLIC_SUPABASE_ANON_KEY"
  );
}

// ── Singleton client (browser + server components) ─────────────────────────
export const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

// ── TypeScript row types (mapped from SQL schema) ──────────────────────────

export interface Trade {
  id: string;
  symbol: string;
  direction: "BUY" | "SELL";
  entry_price: number;
  exit_price: number | null;
  quantity: number;
  stop_loss: number | null;
  target: number | null;
  pnl: number | null;
  agent_name: string;
  strategy: string | null;
  conviction: number | null;
  status: "OPEN" | "CLOSED" | "CANCELLED";
  opened_at: string;
  closed_at: string | null;
  metadata: Record<string, unknown> | null;
}

export interface AgentPerformance {
  id: string;
  agent_name: string;
  date: string;         // ISO date YYYY-MM-DD
  sharpe_30d: number | null;
  win_rate: number | null;
  pnl_total: number | null;
  trades_count: number | null;
  weight: number | null;
}

export interface SignalLog {
  id: string;
  symbol: string;
  signal_score: number | null;
  weights_used: Record<string, number> | null;
  regime: string | null;
  threshold: number | null;
  triggered: boolean | null;
  components: Record<string, number> | null;
  created_at: string;
}

export interface CalibrationLog {
  id: string;
  regime: string | null;
  signal_weights: Record<string, number> | null;
  risk_thresholds: Record<string, number> | null;
  agent_weights: Record<string, number> | null;
  sl_tp_multipliers: Record<string, number> | null;
  position_sizing: Record<string, number> | null;
  kill_switch: boolean;
  reasoning: string | null;
  created_at: string;
}

// ── Convenience query helpers ──────────────────────────────────────────────

/** Open trades sorted newest first. */
export async function fetchOpenTrades(): Promise<Trade[]> {
  const { data, error } = await supabase
    .from("trades")
    .select("*")
    .eq("status", "OPEN")
    .order("opened_at", { ascending: false });
  if (error) throw error;
  return data ?? [];
}

/** Recent closed trades with PnL. */
export async function fetchClosedTrades(limit = 50): Promise<Trade[]> {
  const { data, error } = await supabase
    .from("trades")
    .select("*")
    .eq("status", "CLOSED")
    .order("closed_at", { ascending: false })
    .limit(limit);
  if (error) throw error;
  return data ?? [];
}

/** Latest agent performance rows (one per agent). */
export async function fetchLatestAgentPerformance(): Promise<AgentPerformance[]> {
  // Each agent has one row per day; grab the most recent date available.
  const { data, error } = await supabase
    .from("agent_performance")
    .select("*")
    .order("date", { ascending: false })
    .limit(100);
  if (error) throw error;

  // Deduplicate: keep the latest row per agent.
  const seen = new Set<string>();
  return (data ?? []).filter((row) => {
    if (seen.has(row.agent_name)) return false;
    seen.add(row.agent_name);
    return true;
  });
}

/** Recent signals. */
export async function fetchRecentSignals(limit = 100): Promise<SignalLog[]> {
  const { data, error } = await supabase
    .from("signal_log")
    .select("*")
    .order("created_at", { ascending: false })
    .limit(limit);
  if (error) throw error;
  return data ?? [];
}

/** Most recent calibration runs. */
export async function fetchCalibrationHistory(limit = 20): Promise<CalibrationLog[]> {
  const { data, error } = await supabase
    .from("calibration_log")
    .select("*")
    .order("created_at", { ascending: false })
    .limit(limit);
  if (error) throw error;
  return data ?? [];
}

/** Portfolio KPIs derived from closed trades. */
export async function fetchPortfolioKPIs(): Promise<{
  total_pnl: number;
  win_rate: number;
  open_positions: number;
  today_pnl: number;
}> {
  const today = new Date().toISOString().slice(0, 10);

  const [closedRes, openRes] = await Promise.all([
    supabase.from("trades").select("pnl, closed_at").eq("status", "CLOSED"),
    supabase
      .from("trades")
      .select("id", { count: "exact", head: true })
      .eq("status", "OPEN"),
  ]);

  if (closedRes.error) throw closedRes.error;

  const closed: Pick<Trade, "pnl" | "closed_at">[] = closedRes.data ?? [];
  const total_pnl = closed.reduce((s, r) => s + (r.pnl ?? 0), 0);
  const winners = closed.filter((r) => (r.pnl ?? 0) > 0).length;
  const win_rate = closed.length ? winners / closed.length : 0;
  const today_pnl = closed
    .filter((r) => r.closed_at?.startsWith(today))
    .reduce((s, r) => s + (r.pnl ?? 0), 0);

  return {
    total_pnl,
    win_rate,
    open_positions: openRes.count ?? 0,
    today_pnl,
  };
}
