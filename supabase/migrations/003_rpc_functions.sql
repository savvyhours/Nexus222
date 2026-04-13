-- NEXUS-II — RPC Functions
-- Migration: 003_rpc_functions.sql
--
-- Callable via supabase.rpc('function_name', { params })
-- All financial values in INR (₹).

-- ── 1. Portfolio KPIs (single-call summary) ───────────────────────────────
-- Used by dashboard/lib/supabase.ts → fetchPortfolioKPIs()
CREATE OR REPLACE FUNCTION get_portfolio_kpis(lookback_days INT DEFAULT 30)
RETURNS TABLE (
    total_pnl          DECIMAL,
    today_pnl          DECIMAL,
    win_rate           DECIMAL,
    profit_factor      DECIMAL,
    avg_conviction     DECIMAL,
    total_trades       BIGINT,
    open_positions     BIGINT,
    max_drawdown_pct   DECIMAL
)
LANGUAGE sql STABLE AS $$
    WITH closed AS (
        SELECT
            pnl,
            conviction,
            closed_at
        FROM trades
        WHERE status = 'CLOSED'
          AND closed_at >= NOW() - MAKE_INTERVAL(days => lookback_days)
    ),
    open_cnt AS (
        SELECT COUNT(*) AS cnt FROM trades WHERE status = 'OPEN'
    ),
    equity AS (
        SELECT
            pnl,
            SUM(pnl) OVER (ORDER BY closed_at) AS cum_pnl
        FROM closed
    ),
    drawdown AS (
        SELECT
            MIN(
                cum_pnl - MAX(cum_pnl) OVER (ORDER BY ROW_NUMBER() OVER ())
            ) / NULLIF(MAX(MAX(cum_pnl) OVER ()), 0) * 100 AS max_dd_pct
        FROM equity
    )
    SELECT
        COALESCE(SUM(pnl), 0)                                              AS total_pnl,
        COALESCE(SUM(CASE
            WHEN DATE(closed_at AT TIME ZONE 'Asia/Kolkata') = CURRENT_DATE
            THEN pnl END), 0)                                              AS today_pnl,
        COALESCE(
            COUNT(CASE WHEN pnl > 0 THEN 1 END)::DECIMAL
            / NULLIF(COUNT(*), 0), 0
        )                                                                  AS win_rate,
        COALESCE(
            SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END)
            / NULLIF(ABS(SUM(CASE WHEN pnl < 0 THEN pnl ELSE 0 END)), 0), 0
        )                                                                  AS profit_factor,
        COALESCE(AVG(conviction), 0)                                       AS avg_conviction,
        COUNT(*)                                                           AS total_trades,
        (SELECT cnt FROM open_cnt)                                         AS open_positions,
        COALESCE((SELECT max_dd_pct FROM drawdown), 0)                     AS max_drawdown_pct
    FROM closed;
$$;

-- ── 2. Agent Weight History (time-series for a specific agent) ────────────
CREATE OR REPLACE FUNCTION get_agent_weight_history(
    p_agent_name TEXT,
    p_days       INT DEFAULT 30
)
RETURNS TABLE (
    trade_date DATE,
    weight     DECIMAL,
    sharpe_30d DECIMAL,
    win_rate   DECIMAL,
    pnl_total  DECIMAL
)
LANGUAGE sql STABLE AS $$
    SELECT
        date         AS trade_date,
        weight,
        sharpe_30d,
        win_rate,
        pnl_total
    FROM agent_performance
    WHERE agent_name = p_agent_name
      AND date >= CURRENT_DATE - p_days
    ORDER BY date;
$$;

-- ── 3. Regime Signal Score Distribution ─────────────────────────────────
-- Histogram data: for each regime, how are signal scores distributed?
CREATE OR REPLACE FUNCTION get_signal_score_histogram(
    p_regime  TEXT DEFAULT NULL,
    p_buckets INT  DEFAULT 10
)
RETURNS TABLE (
    bucket_min  DECIMAL,
    bucket_max  DECIMAL,
    regime      TEXT,
    count       BIGINT
)
LANGUAGE sql STABLE AS $$
    WITH bounds AS (
        SELECT MIN(signal_score) AS mn, MAX(signal_score) AS mx
        FROM signal_log
        WHERE (p_regime IS NULL OR regime = p_regime)
    ),
    bucketed AS (
        SELECT
            regime,
            FLOOR(
                (signal_score - b.mn) / NULLIF(b.mx - b.mn, 0) * (p_buckets - 1)
            ) AS bucket_idx
        FROM signal_log, bounds b
        WHERE signal_score IS NOT NULL
          AND (p_regime IS NULL OR signal_log.regime = p_regime)
    ),
    bucket_agg AS (
        SELECT
            regime,
            bucket_idx,
            COUNT(*) AS cnt,
            (SELECT mn FROM bounds) AS global_min,
            (SELECT mx FROM bounds) AS global_max
        FROM bucketed
        GROUP BY regime, bucket_idx
    )
    SELECT
        ROUND((global_min + bucket_idx * (global_max - global_min) / p_buckets)::NUMERIC, 4)       AS bucket_min,
        ROUND((global_min + (bucket_idx + 1) * (global_max - global_min) / p_buckets)::NUMERIC, 4) AS bucket_max,
        regime,
        cnt AS count
    FROM bucket_agg
    ORDER BY regime, bucket_idx;
$$;

-- ── 4. Recent Signals with Trade Outcome ─────────────────────────────────
-- Joins signal_log → trades to show whether a triggered signal was profitable.
CREATE OR REPLACE FUNCTION get_signals_with_outcomes(p_limit INT DEFAULT 50)
RETURNS TABLE (
    signal_id       UUID,
    symbol          TEXT,
    signal_score    DECIMAL,
    regime          TEXT,
    triggered       BOOLEAN,
    threshold       DECIMAL,
    signal_time     TIMESTAMPTZ,
    trade_pnl       DECIMAL,
    trade_direction TEXT,
    trade_status    TEXT
)
LANGUAGE sql STABLE AS $$
    SELECT
        sl.id            AS signal_id,
        sl.symbol,
        sl.signal_score,
        sl.regime,
        sl.triggered,
        sl.threshold,
        sl.created_at    AS signal_time,
        t.pnl            AS trade_pnl,
        t.direction      AS trade_direction,
        t.status         AS trade_status
    FROM signal_log sl
    LEFT JOIN LATERAL (
        SELECT pnl, direction, status
        FROM trades
        WHERE symbol = sl.symbol
          AND opened_at >= sl.created_at - INTERVAL '5 minutes'
          AND opened_at <= sl.created_at + INTERVAL '10 minutes'
        ORDER BY opened_at
        LIMIT 1
    ) t ON TRUE
    ORDER BY sl.created_at DESC
    LIMIT p_limit;
$$;

-- ── 5. Kill-Switch Events ─────────────────────────────────────────────────
-- Dashboard alert panel: when was the kill switch activated?
CREATE OR REPLACE FUNCTION get_kill_switch_events(p_days INT DEFAULT 30)
RETURNS TABLE (
    activated_at TIMESTAMPTZ,
    regime       TEXT,
    reasoning    TEXT
)
LANGUAGE sql STABLE AS $$
    SELECT
        created_at AS activated_at,
        regime,
        LEFT(reasoning, 300) AS reasoning
    FROM calibration_log
    WHERE kill_switch = TRUE
      AND created_at >= NOW() - MAKE_INTERVAL(days => p_days)
    ORDER BY created_at DESC;
$$;

-- ── 6. Equity Curve (for chart.js line chart) ────────────────────────────
-- Returns daily cumulative PnL data points.
CREATE OR REPLACE FUNCTION get_equity_curve(p_days INT DEFAULT 90)
RETURNS TABLE (
    trade_date     DATE,
    daily_pnl      DECIMAL,
    cumulative_pnl DECIMAL,
    trade_count    BIGINT
)
LANGUAGE sql STABLE AS $$
    WITH daily AS (
        SELECT
            DATE(closed_at AT TIME ZONE 'Asia/Kolkata') AS trade_date,
            SUM(pnl)                                    AS daily_pnl,
            COUNT(*)                                    AS trade_count
        FROM trades
        WHERE status = 'CLOSED'
          AND closed_at >= NOW() - MAKE_INTERVAL(days => p_days)
        GROUP BY 1
    )
    SELECT
        trade_date,
        daily_pnl,
        SUM(daily_pnl) OVER (ORDER BY trade_date)  AS cumulative_pnl,
        trade_count
    FROM daily
    ORDER BY trade_date;
$$;

-- ── 7. Component Signal Heatmap ───────────────────────────────────────────
-- Average component values per symbol over last N days — for heatmap chart.
CREATE OR REPLACE FUNCTION get_component_heatmap(p_days INT DEFAULT 7)
RETURNS TABLE (
    symbol              TEXT,
    avg_technical       DECIMAL,
    avg_sentiment       DECIMAL,
    avg_fundamental     DECIMAL,
    avg_macro           DECIMAL,
    avg_candlestick     DECIMAL,
    avg_ml_qlib         DECIMAL,
    avg_debate          DECIMAL,
    signal_count        BIGINT
)
LANGUAGE sql STABLE AS $$
    SELECT
        symbol,
        ROUND(AVG((components->>'technical')::DECIMAL)::NUMERIC, 4)         AS avg_technical,
        ROUND(AVG((components->>'sentiment')::DECIMAL)::NUMERIC, 4)         AS avg_sentiment,
        ROUND(AVG((components->>'fundamental')::DECIMAL)::NUMERIC, 4)       AS avg_fundamental,
        ROUND(AVG((components->>'macro')::DECIMAL)::NUMERIC, 4)             AS avg_macro,
        ROUND(AVG((components->>'candlestick')::DECIMAL)::NUMERIC, 4)       AS avg_candlestick,
        ROUND(AVG((components->>'ml_qlib')::DECIMAL)::NUMERIC, 4)           AS avg_ml_qlib,
        ROUND(AVG((components->>'debate_conviction')::DECIMAL)::NUMERIC, 4) AS avg_debate,
        COUNT(*)                                                             AS signal_count
    FROM signal_log
    WHERE created_at >= NOW() - MAKE_INTERVAL(days => p_days)
      AND components IS NOT NULL
    GROUP BY symbol
    ORDER BY signal_count DESC;
$$;

-- Grant execute to authenticated role (Supabase default)
GRANT EXECUTE ON FUNCTION get_portfolio_kpis(INT)            TO authenticated, anon;
GRANT EXECUTE ON FUNCTION get_agent_weight_history(TEXT,INT) TO authenticated, anon;
GRANT EXECUTE ON FUNCTION get_signal_score_histogram(TEXT,INT) TO authenticated, anon;
GRANT EXECUTE ON FUNCTION get_signals_with_outcomes(INT)     TO authenticated, anon;
GRANT EXECUTE ON FUNCTION get_kill_switch_events(INT)        TO authenticated, anon;
GRANT EXECUTE ON FUNCTION get_equity_curve(INT)              TO authenticated, anon;
GRANT EXECUTE ON FUNCTION get_component_heatmap(INT)         TO authenticated, anon;
