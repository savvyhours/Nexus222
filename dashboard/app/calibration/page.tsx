/**
 * NEXUS-II — Weight Calibration Log
 * Shows calibration_log: each WeightCalibrationAgent run with the full
 * weight JSON, regime, kill-switch status, and LLM reasoning.
 */
import { fetchCalibrationHistory } from "@/lib/supabase";
import type { CalibrationLog } from "@/lib/supabase";

// ── Constants ─────────────────────────────────────────────────────────────

const REGIME_COLORS: Record<string, string> = {
  TRENDING:       "text-blue-400 bg-blue-900/40 border-blue-800",
  MEAN_REVERTING: "text-violet-400 bg-violet-900/40 border-violet-800",
  HIGH_VOL:       "text-orange-400 bg-orange-900/40 border-orange-800",
  LOW_VOL:        "text-teal-400 bg-teal-900/40 border-teal-800",
  CRISIS:         "text-red-400 bg-red-900/40 border-red-800",
};

const SIGNAL_WEIGHT_KEYS = [
  "technical",
  "sentiment",
  "fundamental",
  "macro",
  "candlestick",
  "ml_qlib",
  "debate_conviction",
];

// ── Weight mini-chart ─────────────────────────────────────────────────────

function WeightChart({
  weights,
  keys,
  label,
}: {
  weights: Record<string, number> | null;
  keys: string[];
  label: string;
}) {
  if (!weights) return <span className="text-slate-600 text-xs">—</span>;

  const total = keys.reduce((s, k) => s + Math.abs(weights[k] ?? 0), 0);

  return (
    <div>
      <div className="text-xs text-slate-500 mb-2 font-semibold uppercase tracking-wider">
        {label}
      </div>
      <div className="space-y-1">
        {keys.map((k) => {
          const v = weights[k] ?? 0;
          const barPct = total > 0 ? (Math.abs(v) / total) * 100 : 0;
          return (
            <div key={k} className="flex items-center gap-2">
              <div className="w-24 text-right text-xs text-slate-500 shrink-0 truncate">{k}</div>
              <div className="flex-1 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-indigo-500 rounded-full"
                  style={{ width: `${barPct.toFixed(1)}%` }}
                />
              </div>
              <div className="w-12 text-xs text-slate-400 font-mono text-right">
                {v.toFixed(3)}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── Single calibration card ───────────────────────────────────────────────

function CalibrationCard({ entry, index }: { entry: CalibrationLog; index: number }) {
  const regime = entry.regime ?? "UNKNOWN";
  const regimeClass =
    REGIME_COLORS[regime] ?? "text-slate-400 bg-slate-800/40 border-slate-700";
  const isKillSwitch = entry.kill_switch;

  const agentKeys = entry.agent_weights ? Object.keys(entry.agent_weights).sort() : [];
  const riskKeys = entry.risk_thresholds ? Object.keys(entry.risk_thresholds).sort() : [];

  return (
    <div
      className={`bg-slate-900 border rounded-xl overflow-hidden ${
        isKillSwitch ? "border-red-800" : "border-slate-800"
      }`}
    >
      {/* Card header */}
      <div
        className={`px-5 py-3 border-b ${
          isKillSwitch ? "border-red-800 bg-red-950/30" : "border-slate-800"
        } flex items-center justify-between gap-3 flex-wrap`}
      >
        <div className="flex items-center gap-3">
          <span className="text-slate-500 text-xs font-mono">#{index + 1}</span>
          <span
            className={`text-xs px-2 py-0.5 rounded border font-semibold ${regimeClass}`}
          >
            {regime}
          </span>
          {isKillSwitch && (
            <span className="text-xs px-2 py-0.5 rounded bg-red-900 text-red-300 font-bold border border-red-700">
              KILL SWITCH
            </span>
          )}
        </div>
        <span className="text-slate-500 text-xs">
          {new Date(entry.created_at).toLocaleString("en-IN", {
            month: "short",
            day: "numeric",
            year: "numeric",
            hour: "2-digit",
            minute: "2-digit",
            timeZone: "Asia/Kolkata",
          })}{" "}
          IST
        </span>
      </div>

      <div className="p-5 space-y-6">
        {/* Weight charts — 3 columns */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <WeightChart
            weights={entry.signal_weights}
            keys={SIGNAL_WEIGHT_KEYS}
            label="Signal Weights"
          />
          <WeightChart
            weights={entry.agent_weights}
            keys={agentKeys}
            label="Agent Weights"
          />
          <WeightChart
            weights={entry.risk_thresholds}
            keys={riskKeys}
            label="Risk Thresholds"
          />
        </div>

        {/* SL/TP multipliers + position sizing */}
        {(entry.sl_tp_multipliers || entry.position_sizing) && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {entry.sl_tp_multipliers && (
              <div className="bg-slate-800/50 rounded-lg p-3">
                <div className="text-xs text-slate-500 uppercase tracking-wider mb-2 font-semibold">
                  SL / TP Multipliers
                </div>
                <div className="space-y-1">
                  {Object.entries(entry.sl_tp_multipliers).map(([k, v]) => (
                    <div key={k} className="flex justify-between text-xs">
                      <span className="text-slate-400">{k}</span>
                      <span className="text-slate-300 font-mono">{Number(v).toFixed(3)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
            {entry.position_sizing && (
              <div className="bg-slate-800/50 rounded-lg p-3">
                <div className="text-xs text-slate-500 uppercase tracking-wider mb-2 font-semibold">
                  Position Sizing
                </div>
                <div className="space-y-1">
                  {Object.entries(entry.position_sizing).map(([k, v]) => (
                    <div key={k} className="flex justify-between text-xs">
                      <span className="text-slate-400">{k}</span>
                      <span className="text-slate-300 font-mono">{Number(v).toFixed(3)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* LLM reasoning */}
        {entry.reasoning && (
          <div className="bg-slate-950 border border-slate-800 rounded-lg p-4">
            <div className="text-xs text-slate-500 uppercase tracking-wider mb-2 font-semibold">
              Calibration Reasoning (Claude Sonnet)
            </div>
            <p className="text-slate-300 text-sm leading-relaxed whitespace-pre-wrap">
              {entry.reasoning}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────

export default async function CalibrationPage() {
  const history = await fetchCalibrationHistory(20);

  const killSwitchCount = history.filter((e) => e.kill_switch).length;
  const regimeCounts = history.reduce<Record<string, number>>((acc, e) => {
    const r = e.regime ?? "UNKNOWN";
    acc[r] = (acc[r] ?? 0) + 1;
    return acc;
  }, {});

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-xl font-bold text-white">Weight Calibration Log</h1>
        <p className="text-slate-500 text-sm mt-0.5">
          WeightCalibrationAgent runs — dynamic weights, regime classification, and LLM reasoning
        </p>
      </div>

      {/* Summary */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
          <div className="text-xs text-slate-500 uppercase tracking-wider mb-2">Total Runs</div>
          <div className="text-2xl font-bold text-white">{history.length}</div>
        </div>
        <div
          className={`border rounded-xl p-4 ${
            killSwitchCount > 0
              ? "bg-red-950/30 border-red-800"
              : "bg-slate-900 border-slate-800"
          }`}
        >
          <div className="text-xs text-slate-500 uppercase tracking-wider mb-2">Kill Switches</div>
          <div
            className={`text-2xl font-bold ${killSwitchCount > 0 ? "text-red-400" : "text-white"}`}
          >
            {killSwitchCount}
          </div>
        </div>
        {/* Most frequent regime */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
          <div className="text-xs text-slate-500 uppercase tracking-wider mb-2">Top Regime</div>
          <div className="text-lg font-bold text-indigo-400">
            {Object.entries(regimeCounts).sort((a, b) => b[1] - a[1])[0]?.[0] ?? "—"}
          </div>
        </div>
        {/* Latest */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
          <div className="text-xs text-slate-500 uppercase tracking-wider mb-2">Latest</div>
          <div className="text-sm font-semibold text-white">
            {history[0]
              ? new Date(history[0].created_at).toLocaleString("en-IN", {
                  month: "short",
                  day: "numeric",
                  hour: "2-digit",
                  minute: "2-digit",
                  timeZone: "Asia/Kolkata",
                }) + " IST"
              : "—"}
          </div>
        </div>
      </div>

      {/* Regime frequency mini-table */}
      {Object.keys(regimeCounts).length > 0 && (
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
          <h2 className="font-semibold text-white mb-4">Regime Frequency</h2>
          <div className="flex flex-wrap gap-3">
            {Object.entries(regimeCounts)
              .sort((a, b) => b[1] - a[1])
              .map(([regime, count]) => {
                const cls = REGIME_COLORS[regime] ?? "text-slate-400 bg-slate-800 border-slate-700";
                return (
                  <div
                    key={regime}
                    className={`px-3 py-1.5 rounded-lg border text-sm font-semibold ${cls}`}
                  >
                    {regime}{" "}
                    <span className="font-mono ml-1">×{count}</span>
                  </div>
                );
              })}
          </div>
        </div>
      )}

      {/* Calibration cards */}
      {history.length === 0 ? (
        <div className="bg-slate-900 border border-slate-800 rounded-xl py-16 text-center">
          <p className="text-slate-600 text-sm">
            No calibration runs yet — WeightCalibrationAgent will populate this log once the bot
            starts
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {history.map((entry, i) => (
            <CalibrationCard key={entry.id} entry={entry} index={i} />
          ))}
        </div>
      )}
    </div>
  );
}
