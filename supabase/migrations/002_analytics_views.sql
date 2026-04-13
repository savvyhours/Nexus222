-- NEXUS-II — Analytics Views
-- Migration: 002_analytics_views.sql
-- These views power the Next.js dashboard pages.
-- All monetary values in INR (₹).

-- ── 1. Daily PnL Summary ─────────────────────────────────────────────────
-- Used by Portfolio overview KPI cards.
CREATE OR REPLACE VIEW v_daily_pnl AS
SELECT
    DATE(closed_at AT TIME ZONE 'Asia/Kolkata')   AS trade_date,
    COUNT(*)                                       AS total_trades,
    SUM(pnl)                                       AS total_pnl,
    SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END)    AS gross_profit,
    SUM(CASE WHEN pnl < 0 THEN pnl ELSE 0 END)    AS gross_loss,
    COUNT(CASE WHEN pnl > 0 THEN 1 END)            AS winners,
    COUNT(CASE WHEN pnl < 0 THEN 1 END)            AS losers,
    ROUND(
        COUNT(CASE WHEN pnl > 0 THEN 1 END)::NUMERIC
        / NULLIF(COUNT(*), 0) * 100, 2
    )                                              AS win_rate_pct,
    AVG(conviction)                                AS avg_conviction
FROM trades
WHERE status = 'CLOSED'
  AND closed_at IS NOT NULL
GROUP BY 1
ORDER BY 1 DESC;

-- ── 2. Agent Leaderboard ──────────────────────────────────────────────────
-- Latest performance row per agent with 30d rolling stats.
CREATE OR REPLACE VIEW v_agent_leaderboard AS
WITH latest_per_agent AS (
    SELECT DISTINCT ON (agent_name)
        agent_name,
        date,
        sharpe_30d,
        win_rate,
        pnl_total,
        trades_count,
        weight
    FROM agent_performance
    ORDER BY agent_name, date DESC
),
trade_stats AS (
    SELECT
        agent_name,
        COUNT(*)                                               AS lifetime_trades,
        SUM(pnl)                                               AS lifetime_pnl,
        ROUND(
            COUNT(CASE WHEN pnl > 0 THEN 1 END)::NUMERIC
            / NULLIF(COUNT(*), 0) * 100, 2
        )                                                      AS lifetime_win_pct,
        AVG(CASE WHEN status = 'CLOSED' THEN pnl END)         AS avg_pnl_per_trade,
        -- Profit Factor = gross_profit / |gross_loss|
        ROUND(
            SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END)
            / NULLIF(ABS(SUM(CASE WHEN pnl < 0 THEN pnl ELSE 0 END)), 0), 2
        )                                                      AS profit_factor
    FROM trades
    WHERE status = 'CLOSED'
    GROUP BY agent_name
)
SELECT
    l.agent_name,
    l.date             AS as_of_date,
    l.sharpe_30d,
    l.win_rate         AS win_rate_30d,
    l.pnl_total        AS pnl_30d,
    l.trades_count     AS trades_30d,
    l.weight           AS current_weight,
    t.lifetime_trades,
    t.lifetime_pnl,
    t.lifetime_win_pct,
    t.avg_pnl_per_trade,
    t.profit_factor,
    -- Rank by weight desc, then Sharpe
    ROW_NUMBER() OVER (ORDER BY l.weight DESC NULLS LAST, l.sharpe_30d DESC NULLS LAST) AS rank
FROM latest_per_agent l
LEFT JOIN trade_stats t ON t.agent_name = l.agent_name
ORDER BY rank;

-- ── 3. Signal Statistics ──────────────────────────────────────────────────
-- Aggregate signal outcomes by regime and symbol.
CREATE OR REPLACE VIEW v_signal_stats AS
SELECT
    regime,
    symbol,
    COUNT(*)                                                    AS total_signals,
    COUNT(CASE WHEN triggered THEN 1 END)                       AS triggered_count,
    ROUND(
        COUNT(CASE WHEN triggered THEN 1 END)::NUMERIC
        / NULLIF(COUNT(*), 0) * 100, 2
    )                                                           AS trigger_rate_pct,
    ROUND(AVG(signal_score)::NUMERIC, 4)                        AS avg_score,
    ROUND(MAX(signal_score)::NUMERIC, 4)                        AS max_score,
    ROUND(MIN(signal_score)::NUMERIC, 4)                        AS min_score,
    ROUND(AVG(threshold)::NUMERIC, 4)                           AS avg_threshold,
    MAX(created_at)                                             AS last_signal_at
FROM signal_log
GROUP BY regime, symbol
ORDER BY total_signals DESC;

-- ── 4. Open Positions with Unrealised PnL ────────────────────────────────
-- Joins trades with latest market price (placeholder column).
-- Real impl: join with a prices table or compute via DhanHQ API.
CREATE OR REPLACE VIEW v_open_positions AS
SELECT
    t.id,
    t.symbol,
    t.direction,
    t.entry_price,
    t.quantity,
    t.stop_loss,
    t.target,
    t.agent_name,
    t.strategy,
    t.conviction,
    t.opened_at AT TIME ZONE 'Asia/Kolkata' AS opened_at_ist,
    -- Placeholder: unrealised_pnl populated by application layer
    NULL::DECIMAL AS current_price,
    NULL::DECIMAL AS unrealised_pnl,
    EXTRACT(EPOCH FROM (NOW() - t.opened_at)) / 60 AS age_minutes
FROM trades t
WHERE t.status = 'OPEN'
ORDER BY t.opened_at DESC;

-- ── 5. Calibration Regime History ────────────────────────────────────────
-- Time-series of regime changes detected by WeightCalibrationAgent.
CREATE OR REPLACE VIEW v_regime_history AS
SELECT
    id,
    regime,
    kill_switch,
    -- Extract key signal weights for quick display
    (signal_weights->>'technical')::DECIMAL        AS w_technical,
    (signal_weights->>'sentiment')::DECIMAL        AS w_sentiment,
    (signal_weights->>'fundamental')::DECIMAL      AS w_fundamental,
    (signal_weights->>'macro')::DECIMAL            AS w_macro,
    (signal_weights->>'candlestick')::DECIMAL      AS w_candlestick,
    (signal_weights->>'ml_qlib')::DECIMAL          AS w_ml_qlib,
    (signal_weights->>'debate_conviction')::DECIMAL AS w_debate,
    (risk_thresholds->>'signal_threshold')::DECIMAL AS signal_threshold,
    LEFT(reasoning, 200)                           AS reasoning_excerpt,
    created_at AT TIME ZONE 'Asia/Kolkata'         AS created_at_ist
FROM calibration_log
ORDER BY created_at DESC;

-- ── 6. PnL Cumulative (running total) ────────────────────────────────────
-- Used for the equity curve chart on the dashboard.
CREATE OR REPLACE VIEW v_equity_curve AS
SELECT
    closed_at AT TIME ZONE 'Asia/Kolkata'          AS closed_at_ist,
    symbol,
    direction,
    pnl,
    SUM(pnl) OVER (ORDER BY closed_at ROWS UNBOUNDED PRECEDING) AS cumulative_pnl,
    agent_name
FROM trades
WHERE status = 'CLOSED'
  AND closed_at IS NOT NULL
ORDER BY closed_at;

-- ── 7. Strategy Performance Breakdown ────────────────────────────────────
-- Per-strategy PnL and hit rate.
CREATE OR REPLACE VIEW v_strategy_performance AS
SELECT
    strategy,
    agent_name,
    COUNT(*)                                                   AS trades,
    SUM(pnl)                                                   AS total_pnl,
    ROUND(AVG(pnl)::NUMERIC, 2)                                AS avg_pnl,
    COUNT(CASE WHEN pnl > 0 THEN 1 END)                        AS winners,
    ROUND(
        COUNT(CASE WHEN pnl > 0 THEN 1 END)::NUMERIC
        / NULLIF(COUNT(*), 0) * 100, 2
    )                                                          AS win_rate_pct,
    ROUND(AVG(conviction)::NUMERIC, 3)                         AS avg_conviction
FROM trades
WHERE status = 'CLOSED'
  AND strategy IS NOT NULL
GROUP BY strategy, agent_name
ORDER BY total_pnl DESC;

-- ── Indexes for view performance ─────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_trades_closed_at    ON trades(closed_at DESC)  WHERE status = 'CLOSED';
CREATE INDEX IF NOT EXISTS idx_trades_agent_status ON trades(agent_name, status);
CREATE INDEX IF NOT EXISTS idx_signal_log_regime   ON signal_log(regime, symbol);
CREATE INDEX IF NOT EXISTS idx_cal_log_created     ON calibration_log(created_at DESC);
