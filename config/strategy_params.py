"""
NEXUS-II — Strategy Parameters (Static Defaults)
Dynamic values are always overridden by WeightCalibrationAgent at runtime.
These are fallback/initial values only.
"""

# ── Signal Engine (defaults — overridden by WeightCalibrationAgent) ───────
DEFAULT_SIGNAL_WEIGHTS = {
    "technical": 0.25,
    "sentiment": 0.15,
    "fundamental": 0.15,
    "macro": 0.10,
    "candlestick": 0.10,
    "ml_qlib": 0.15,
    "debate_conviction": 0.10,
}
DEFAULT_SIGNAL_THRESHOLD = 0.60

# ── Agent Weights (defaults) ──────────────────────────────────────────────
DEFAULT_AGENT_WEIGHTS = {
    "scalper": 0.10,
    "trend_follower": 0.12,
    "options": 0.10,
    "mean_reversion": 0.10,
    "sentiment": 0.10,
    "fundamentals": 0.10,
    "macro": 0.10,
    "pattern": 0.08,
    "quant": 0.12,
    "etf": 0.08,
}

# ── Risk Defaults ─────────────────────────────────────────────────────────
DEFAULT_RISK_THRESHOLDS = {
    "max_position_pct": 0.05,          # 5% of capital per stock
    "max_sector_pct": 0.25,            # 25% in one sector
    "max_daily_loss_pct": 0.02,        # 2% daily loss → halt
    "max_drawdown_pct": 0.08,          # 8% drawdown → pause
    "vix_defensive_threshold": 22,     # VIX > 22 → defensive
    "vix_halt_threshold": 28,          # VIX > 28 → halt all
    "min_liquidity_volume": 50_000,    # Min 50K volume/min
    "min_fno_oi": 500_000,             # Min OI for FnO trades
    "news_blackout_minutes": 30,       # Blackout before/after events
    "correlation_max": 0.80,           # Max correlation between positions
    "margin_buffer_pct": 0.20,         # Keep 20% margin buffer
}

# ── SL/TP Multipliers ─────────────────────────────────────────────────────
DEFAULT_SL_TP = {
    "intraday_sl_atr": 2.0,
    "positional_sl_atr": 3.0,
    "trailing_stop_atr": 1.5,
    "target_risk_reward": 2.0,
}

# ── Position Sizing ───────────────────────────────────────────────────────
DEFAULT_POSITION_SIZING = {
    "default_pct": 0.03,
    "max_pct": 0.05,
    "high_conviction_pct": 0.05,
    "low_conviction_pct": 0.01,
}

# ── Scalper Params ────────────────────────────────────────────────────────
SCALPER_MAX_HOLD_MINUTES = 30
SCALPER_RSI_PERIOD = 14
SCALPER_EMA_FAST = 9
SCALPER_EMA_SLOW = 21
SCALPER_VOLUME_SPIKE_MULTIPLIER = 2.0

# ── Trend Follower Params ─────────────────────────────────────────────────
TREND_EMA_SHORT = 20
TREND_EMA_LONG = 50
TREND_ADX_MIN = 25
TREND_52W_HIGH_LOOKBACK = 252

# ── Options Params ────────────────────────────────────────────────────────
OPTIONS_IV_CRUSH_PERCENTILE = 80
OPTIONS_MIN_OI = 500_000
OPTIONS_VIX_HEDGE_TRIGGER = 18
OPTIONS_DELTA_NEUTRAL_MAX = 0.30

# ── Mean Reversion Params ─────────────────────────────────────────────────
MEAN_REV_ZSCORE_THRESHOLD = 2.0
MEAN_REV_BB_PERIOD = 20
MEAN_REV_BB_STD = 2.0

# ── Sentiment Params ─────────────────────────────────────────────────────
SENTIMENT_BUY_THRESHOLD = 0.6
SENTIMENT_SELL_THRESHOLD = -0.6
SENTIMENT_HOLD_HOURS = 2
SENTIMENT_SWING_THRESHOLD = 0.4
SENTIMENT_SWING_MINUTES = 30
