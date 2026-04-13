-- NEXUS-II — Initial Supabase Schema
-- Migration: 001_initial_schema.sql

-- ── Trades ────────────────────────────────────────────────────────────────
CREATE TABLE trades (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('BUY', 'SELL')),
    entry_price DECIMAL NOT NULL,
    exit_price DECIMAL,
    quantity INT NOT NULL,
    stop_loss DECIMAL,
    target DECIMAL,
    pnl DECIMAL,
    agent_name TEXT NOT NULL,
    strategy TEXT,
    conviction DECIMAL,
    status TEXT DEFAULT 'OPEN' CHECK (status IN ('OPEN', 'CLOSED', 'CANCELLED')),
    opened_at TIMESTAMPTZ DEFAULT NOW(),
    closed_at TIMESTAMPTZ,
    metadata JSONB
);

-- ── Agent Performance ─────────────────────────────────────────────────────
CREATE TABLE agent_performance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_name TEXT NOT NULL,
    date DATE NOT NULL,
    sharpe_30d DECIMAL,
    win_rate DECIMAL,
    pnl_total DECIMAL,
    trades_count INT,
    weight DECIMAL,
    UNIQUE(agent_name, date)
);

-- ── Signal Log ────────────────────────────────────────────────────────────
CREATE TABLE signal_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol TEXT NOT NULL,
    signal_score DECIMAL,
    weights_used JSONB,
    regime TEXT,
    threshold DECIMAL,
    triggered BOOLEAN,
    components JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ── Weight Calibration Log ────────────────────────────────────────────────
CREATE TABLE calibration_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    regime TEXT,
    signal_weights JSONB,
    risk_thresholds JSONB,
    agent_weights JSONB,
    sl_tp_multipliers JSONB,
    position_sizing JSONB,
    kill_switch BOOLEAN DEFAULT FALSE,
    reasoning TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ── Sentiment Cache ───────────────────────────────────────────────────────
CREATE TABLE sentiment_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol TEXT NOT NULL,
    score DECIMAL,
    source TEXT,
    tier INT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ── Factor Scores (Qlib) ──────────────────────────────────────────────────
CREATE TABLE factor_scores (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    market TEXT NOT NULL,
    symbol TEXT NOT NULL,
    date DATE NOT NULL,
    predicted_return DECIMAL,
    rank INT,
    factors JSONB,
    UNIQUE(market, symbol, date)
);

-- ── Enable Realtime ───────────────────────────────────────────────────────
ALTER PUBLICATION supabase_realtime ADD TABLE trades;
ALTER PUBLICATION supabase_realtime ADD TABLE agent_performance;

-- ── Indexes ───────────────────────────────────────────────────────────────
CREATE INDEX idx_trades_symbol ON trades(symbol);
CREATE INDEX idx_trades_status ON trades(status);
CREATE INDEX idx_trades_opened_at ON trades(opened_at DESC);
CREATE INDEX idx_signal_log_symbol ON signal_log(symbol);
CREATE INDEX idx_signal_log_created_at ON signal_log(created_at DESC);
CREATE INDEX idx_factor_scores_market_date ON factor_scores(market, date DESC);
CREATE INDEX idx_sentiment_cache_symbol ON sentiment_cache(symbol, created_at DESC);
