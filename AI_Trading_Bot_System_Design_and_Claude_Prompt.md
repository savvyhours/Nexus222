# AI-Powered Automated Trading System for Indian Markets — v2.1
## Complete System Architecture, Design Blueprint & Claude Code Prompt
### By Sourav Saha Roy | Hyderabad, India | April 2026

---

## CHANGELOG v2.0 → v2.1

| Area | v2.0 | v2.1 Change | Rationale |
|------|------|-------------|-----------|
| Risk Management | 12-gate pipeline (new design) | **Restored v1.0 risk system** (PreTradeRiskChecker + PositionManager + DrawdownMonitor) **PLUS** dynamic gate thresholds via WeightCalibrationAgent | User request: keep original risk; add dynamic adaptivity |
| Strategies | TradingAgents-style debate only | **Restored all 10 original sub-agents** (Scalper, Trend, Options, MeanReversion, Sentiment, Fundamental, Macro, Pattern, Quant, ETF) **PLUS** 22 backtested strategies + candlestick composites | User request: keep previous strategies + add new ones |
| LLM for Debate | Claude Opus | **Claude Sonnet** for debate synthesis too | User request: Sonnet for debate |
| Deployment | Docker only | **GitHub + Supabase (DB/Auth) + Cloudflare Workers (API) + Vercel (Dashboard)** | User request: cloud deployment |
| Qlib Adaptation | NSE500 only | **Multi-market factor models**: NSE500, BSE, NIFTY50, NIFTYIT, sector-specific | User request: different adaptability per market |
| Signal Engine | Fixed weights | **Dynamic weights** via WeightCalibrationAgent — adapts to market regime | User request: dynamic based on market conditions |
| Risk Gates | Fixed thresholds | **Dynamic thresholds** via WeightCalibrationAgent — adjusts per VIX/regime | User request: dynamic risk |
| Options Chain Rate Limit | 1 per 3 seconds | **1 per 150 seconds** | User correction |
| All Weights | Hardcoded | **Dynamic WeightCalibrationAgent** — called every time any weight is needed | User request: dedicated skill+agent for all weights |

---

## PART 1: SYSTEM OVERVIEW

This document outlines the full architecture of **NEXUS-II** — an AI-powered, self-learning, multi-agent automated trading system designed specifically for Indian markets (NSE/BSE) covering Equity Intraday, Positional, FnO (Options), and ETFs.

The system integrates patterns from:

| Source | What We Take | How We Adapt |
|--------|-------------|--------------|
| **TauricResearch/TradingAgents** | 4-tier agent hierarchy: Analysts → Researchers → Debate → Portfolio Manager | Use Claude Sonnet for ALL tiers (cost-efficient); replace FinnHub with DhanHQ + Indian data |
| **HKUDS/AI-Trader** | MCP tool-driven architecture; zero pre-programmed strategies; tool-only execution | Build MCP tools wrapping DhanHQ v2 API; agents reason through tools |
| **666ghj/MiroFish** | Swarm simulation with emergent consensus; agent personas with memory | Weekend scenario simulation: inject macro seeds, simulate 100+ agents, extract conviction |
| **microsoft/qlib** | Alpha158 factor library + LightGBM ranking model | **Multi-market adaptation**: separate factor models for NSE500, NIFTY50, NIFTYIT, BSE, sector-level |
| **AI4Finance-Foundation/FinRL** | PPO/SAC ensemble RL for portfolio allocation | Optional Layer — only after 3 months of live data |
| **dhan-oss/DhanHQ-py** | Full broker execution: orders, Super Orders, Conditional Triggers, Kill Switch | Primary execution layer |
| **TheSnowGuru/Market-Swarm-Agents** | Master/Sub-agent hierarchy with Sharpe-weighted voting | **All 10 original sub-agents preserved** + TradingAgents debate overlay |
| **QuantConnect/Lean** | Walk-forward backtesting methodology | vectorbt for speed; Lean methodology for validation discipline |
| **vnpy/vnpy** | Event-driven trading engine pattern | Event bus architecture for signal→risk→execution pipeline |
| **The-Swarm-Corporation/awesome-financial-agents** | Agent design patterns catalog | Reference for agent interface contracts |
| **mvanhorn/last30days-skill** | Rolling performance tracking | 30-day rolling Sharpe for agent weight updates |
| **NEXUS v1 (existing bot)** | 11-agent system, 9 risk gates, signal engine, MiroFish integration, Telegram approval | Foundation — everything proven in paper trading carries forward |

---

## PART 2: SYSTEM ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         NEXUS-II TRADING SYSTEM v2.1                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ LAYER 0: DATA INGESTION (MCP Tool-Driven)                              │ │
│  │  DhanHQ v2 · Finance Intel DB · OpenBB/FRED/AlphaVantage · NSE/BSE    │ │
│  │  News RSS · Screener.in · India VIX · FII/DII Flows                    │ │
│  └────────────────────────────┬───────────────────────────────────────────┘ │
│                               ▼                                             │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ ★ WEIGHT CALIBRATION AGENT (Called before EVERY weight decision)       │ │
│  │  • Reads: VIX, regime, drawdown, win rates, sector momentum, volume   │ │
│  │  • Outputs: signal_weights, risk_thresholds, agent_weights, SL/TP     │ │
│  │  • LLM: Claude Sonnet · Cached: 15-min TTL during market hours        │ │
│  │  • Skill: /skills/weight-calibration/ (see PART 13)                   │ │
│  └────────────────────────────┬───────────────────────────────────────────┘ │
│                               ▼                                             │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ LAYER 1: ANALYST AGENTS (Tier 1 — Data Interpretation)                │ │
│  │  Technical Analyst · Sentiment Analyst · Fundamental Analyst           │ │
│  │  Macro/Flow Analyst (all Claude Sonnet)                                │ │
│  └────────────────────────────┬───────────────────────────────────────────┘ │
│                               ▼                                             │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ LAYER 2: 10 STRATEGY SUB-AGENTS (Original v1.0 Swarm — Preserved)    │ │
│  │  ScalperAgent · TrendFollowerAgent · OptionsAgent · MeanReversionAgent│ │
│  │  SentimentAgent · FundamentalsAgent · MacroAgent · PatternAgent        │ │
│  │  QuantAgent · ETFAgent                                                 │ │
│  │  + 22 Backtested Strategies + Candlestick Composites                  │ │
│  │  Weights: DYNAMIC via WeightCalibrationAgent (30-day Sharpe-based)    │ │
│  └────────────────────────────┬───────────────────────────────────────────┘ │
│                               ▼                                             │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ LAYER 3: RESEARCH AGENTS (Tier 2 — Cross-Domain Synthesis)            │ │
│  │  Bull Researcher (Sonnet) · Bear Researcher (Sonnet)                  │ │
│  │  Risk Researcher (Sonnet)                                              │ │
│  └────────────────────────────┬───────────────────────────────────────────┘ │
│                               ▼                                             │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ LAYER 4: DEBATE ARENA (Tier 3 — Claude SONNET, NOT Opus)              │ │
│  │  Bull vs Bear structured debate (2-3 rounds)                          │ │
│  │  Risk Researcher provides guardrails                                   │ │
│  │  Sonnet synthesizes final conviction score                             │ │
│  └────────────────────────────┬───────────────────────────────────────────┘ │
│                               ▼                                             │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ LAYER 5: PORTFOLIO MANAGER (Tier 4 — Claude Sonnet)                   │ │
│  │  Approve / Reject / Modify each trade proposal                        │ │
│  │  Portfolio-level diversification, correlation, capital allocation      │ │
│  └────────────────────────────┬───────────────────────────────────────────┘ │
│                               ▼                                             │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ LAYER 6: SIGNAL ENGINE (★ DYNAMIC Weights via WeightCalibrationAgent) │ │
│  │  7-Component Scorer — weights CHANGE with market regime:              │ │
│  │  TRENDING: Technical↑ Momentum↑ Sentiment↓                           │ │
│  │  MEAN-REVERTING: MeanReversion↑ Bollinger↑ Technical↓                │ │
│  │  HIGH-VOL: Risk↑ VIX↑ Position-size↓ Options↑                       │ │
│  │  LOW-VOL: Theta-decay↑ Position-size↑ Scalping↑                     │ │
│  │  Threshold: ALSO dynamic (tighter in high-vol, looser in trending)    │ │
│  └────────────────────────────┬───────────────────────────────────────────┘ │
│                               ▼                                             │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ LAYER 7: RISK MANAGEMENT (Original v1.0 — Preserved + Dynamic Gates) │ │
│  │  PreTradeRiskChecker (5 checks) + PositionManager (6 exit triggers)   │ │
│  │  DrawdownMonitor (4 circuit breakers) + Dhan Kill Switch/P&L Exit     │ │
│  │  ALL thresholds: DYNAMIC via WeightCalibrationAgent                   │ │
│  └────────────────────────────┬───────────────────────────────────────────┘ │
│                               ▼                                             │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ LAYER 8: HUMAN-IN-THE-LOOP (Telegram Approval)                        │ │
│  │  ✅ Approve | ❌ Reject | ✏️ Modify                                    │ │
│  │  Auto-approve after 5 min in PAPER_TRADE; never in LIVE               │ │
│  └────────────────────────────┬───────────────────────────────────────────┘ │
│                               ▼                                             │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ LAYER 9: EXECUTION ENGINE (DhanHQ v2 MCP Tools)                       │ │
│  │  Super Orders · Conditional Triggers · Slicing · Margin Calc          │ │
│  │  Paper Trader (simulated) · WebSocket monitoring                       │ │
│  └────────────────────────────┬───────────────────────────────────────────┘ │
│                               ▼                                             │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ LAYER 10: MONITORING & DEPLOYMENT                                     │ │
│  │  Telegram · Streamlit (Vercel) · Guard Watchdog                       │ │
│  │  GitHub (source) + Supabase (DB/Auth) + Cloudflare Workers (API)      │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ WEEKEND: MiroFish Scenario Simulation                                  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## PART 3: THE WEIGHT CALIBRATION AGENT & SKILL (★ NEW — Core Innovation)

**Purpose:** Every single weight, threshold, and parameter in the system is dynamic. Instead of hardcoded values, a dedicated `WeightCalibrationAgent` is called whenever any component needs a weight. This agent reads current market conditions and returns calibrated values.

### 3.1 What It Controls

| Component | What Gets Calibrated | Example Adaptation |
|-----------|---------------------|-------------------|
| Signal Engine weights | 7 component weights (must sum to 1.0) | Trending market → Technical weight ↑ to 0.35, Sentiment ↓ to 0.08 |
| Signal Engine threshold | Consensus threshold (0.45-0.80) | High VIX → threshold tightens to 0.75 (fewer trades, higher conviction) |
| Agent weights | 10 sub-agent vote weights | Agent with -ve Sharpe last 30 days → weight = 0.0 (muted) |
| Risk gate thresholds | Position size %, sector %, daily loss %, drawdown % | VIX > 22 → max position 2.5% (halved), daily loss limit 1% (tighter) |
| Stop-loss multipliers | ATR multiplier for SL/TP | High-vol → SL = 3x ATR (wider), Low-vol → SL = 1.5x ATR (tighter) |
| Position sizing | Default position size % | Drawdown > 4% → position size halved |
| Options parameters | IV percentile threshold, delta limits | IV crush season → IV threshold = 70th (lower bar for theta plays) |
| Qlib model selection | Which factor model to use per market/sector | NIFTYIT → IT-sector factor model; NIFTY50 → broad market model |

### 3.2 WeightCalibrationAgent Implementation

```python
class WeightCalibrationAgent:
    """
    Central brain for ALL dynamic weights in the system.
    Called by every component before using any weight/threshold.
    
    Uses Claude Sonnet to reason about market conditions and output
    calibrated parameters. Results cached for 15 minutes during
    market hours to avoid excessive LLM calls.
    """
    
    CACHE_TTL_MARKET_HOURS = 900   # 15 minutes
    CACHE_TTL_OFF_HOURS = 3600     # 1 hour
    
    def __init__(self, claude_client, market_data_tools, config):
        self.claude = claude_client
        self.tools = market_data_tools
        self.config = config
        self._cache = {}
        self._cache_timestamp = None
    
    async def get_signal_weights(self) -> dict:
        """Returns 7-component signal engine weights (sum to 1.0)."""
        cal = await self._get_calibration()
        return cal["signal_weights"]
    
    async def get_signal_threshold(self) -> float:
        """Returns dynamic consensus threshold (0.45-0.80)."""
        cal = await self._get_calibration()
        return cal["signal_threshold"]
    
    async def get_agent_weights(self) -> dict:
        """Returns weight for each of the 10 sub-agents."""
        cal = await self._get_calibration()
        return cal["agent_weights"]
    
    async def get_risk_thresholds(self) -> dict:
        """Returns dynamic risk gate thresholds."""
        cal = await self._get_calibration()
        return cal["risk_thresholds"]
    
    async def get_sl_tp_multipliers(self) -> dict:
        """Returns ATR multipliers for stop-loss and take-profit."""
        cal = await self._get_calibration()
        return cal["sl_tp_multipliers"]
    
    async def get_position_sizing(self) -> dict:
        """Returns position sizing parameters."""
        cal = await self._get_calibration()
        return cal["position_sizing"]
    
    async def get_qlib_model_config(self, market: str) -> dict:
        """Returns which Qlib factor model to use for a given market/sector."""
        cal = await self._get_calibration()
        return cal["qlib_models"].get(market, cal["qlib_models"]["default"])
    
    async def _get_calibration(self) -> dict:
        """Core method: check cache, if stale → call LLM."""
        now = time.time()
        ttl = self.CACHE_TTL_MARKET_HOURS if self._is_market_hours() else self.CACHE_TTL_OFF_HOURS
        
        if self._cache and self._cache_timestamp and (now - self._cache_timestamp) < ttl:
            return self._cache
        
        # Gather current market state
        market_state = await self._gather_market_state()
        
        # Call Claude Sonnet for calibration
        response = await self.claude.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            system=WEIGHT_CALIBRATION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": self._build_calibration_prompt(market_state)}]
        )
        
        calibration = json.loads(response.content[0].text)
        
        # Validate: signal weights must sum to 1.0
        sw = calibration["signal_weights"]
        total = sum(sw.values())
        calibration["signal_weights"] = {k: v/total for k, v in sw.items()}
        
        # Validate: all thresholds within safe bounds
        calibration = self._enforce_safety_bounds(calibration)
        
        self._cache = calibration
        self._cache_timestamp = now
        return calibration
    
    async def _gather_market_state(self) -> dict:
        """Collect all data the calibration agent needs to reason."""
        return {
            "india_vix": await self.tools.get_india_vix(),
            "nifty_change_pct": await self.tools.get_nifty_change(),
            "fii_dii_flow": await self.tools.get_fii_dii(),
            "market_regime": await self._detect_regime(),  # TRENDING/MEAN_REVERTING/HIGH_VOL/LOW_VOL
            "current_drawdown": self.tools.get_current_drawdown(),
            "daily_pnl_pct": self.tools.get_daily_pnl_pct(),
            "agent_sharpe_30d": self.tools.get_agent_sharpe_scores(),
            "sector_momentum": await self.tools.get_sector_momentum(),
            "market_breadth": await self.tools.get_advance_decline_ratio(),
            "options_iv_percentile": await self.tools.get_avg_iv_percentile(),
            "time_of_day": datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%H:%M"),
            "day_of_week": datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%A"),
        }
    
    def _enforce_safety_bounds(self, cal: dict) -> dict:
        """Hard safety limits that the LLM cannot override."""
        rt = cal["risk_thresholds"]
        rt["max_position_pct"] = min(rt.get("max_position_pct", 0.05), 0.10)      # NEVER > 10%
        rt["max_sector_pct"] = min(rt.get("max_sector_pct", 0.25), 0.35)           # NEVER > 35%
        rt["max_daily_loss_pct"] = min(rt.get("max_daily_loss_pct", 0.02), 0.03)   # NEVER > 3%
        rt["max_drawdown_pct"] = min(rt.get("max_drawdown_pct", 0.08), 0.12)       # NEVER > 12%
        cal["signal_threshold"] = max(min(cal.get("signal_threshold", 0.60), 0.80), 0.40)
        return cal
```

### 3.3 Weight Calibration System Prompt

```
You are the Weight Calibration Agent for NEXUS-II, an Indian market trading system.

Your job: Given the current market state, output calibrated weights and thresholds 
that maximize risk-adjusted returns while preserving capital.

== MARKET REGIMES ==
TRENDING: ADX > 25, clear directional move, low mean-reversion
  → Boost: Technical, Trend, Momentum
  → Reduce: MeanReversion, Scalper
  → Threshold: Moderate (0.55-0.65)
  
MEAN_REVERTING: ADX < 20, range-bound, high mean-reversion
  → Boost: MeanReversion, Bollinger, RSI
  → Reduce: TrendFollower, Momentum
  → Threshold: Standard (0.60)

HIGH_VOL: VIX > 20, large intraday ranges, fear regime
  → Boost: Risk gates (tighten ALL), Options (hedging), VIX plays
  → Reduce: Position sizes (halve), Scalper (too risky), new longs
  → Threshold: Tight (0.70-0.80) — only high-conviction trades
  → SL: Widen (3x ATR) to avoid noise stops

LOW_VOL: VIX < 14, small ranges, complacency
  → Boost: Theta-decay, Scalper (narrow ranges work), Position sizes
  → Reduce: Options (low premium), Hedging (unnecessary)
  → Threshold: Looser (0.50-0.55) — more trades OK
  → SL: Tighter (1.5x ATR)

CRISIS: VIX > 28, circuit breaker territory
  → ALL trading halted, exits only
  → Return kill_switch: true

== AGENT PERFORMANCE RULES ==
- Agent with rolling 30-day Sharpe < 0 → weight = 0.0 (muted)
- Agent with rolling 30-day Sharpe > 2.0 → weight = max(normalized)
- New agent (< 30 days history) → weight = 1/N (equal weight)

== OUTPUT FORMAT ==
Return ONLY valid JSON:
{
  "market_regime": "TRENDING|MEAN_REVERTING|HIGH_VOL|LOW_VOL|CRISIS",
  "signal_weights": {
    "technical": 0.25,
    "sentiment": 0.15,
    "fundamental": 0.15,
    "macro": 0.10,
    "candlestick": 0.10,
    "ml_qlib": 0.15,
    "debate_conviction": 0.10
  },
  "signal_threshold": 0.60,
  "agent_weights": {
    "scalper": 0.10, "trend_follower": 0.12, "options": 0.10,
    "mean_reversion": 0.10, "sentiment": 0.10, "fundamentals": 0.10,
    "macro": 0.10, "pattern": 0.08, "quant": 0.12, "etf": 0.08
  },
  "risk_thresholds": {
    "max_position_pct": 0.05,
    "max_sector_pct": 0.25,
    "max_daily_loss_pct": 0.02,
    "max_drawdown_pct": 0.08,
    "vix_defensive_threshold": 22,
    "vix_halt_threshold": 28,
    "min_liquidity_volume": 50000,
    "min_fno_oi": 500000,
    "news_blackout_minutes": 30,
    "correlation_max": 0.80,
    "margin_buffer_pct": 0.20
  },
  "sl_tp_multipliers": {
    "intraday_sl_atr": 2.0,
    "positional_sl_atr": 3.0,
    "trailing_stop_atr": 1.5,
    "target_risk_reward": 2.0
  },
  "position_sizing": {
    "default_pct": 0.03,
    "max_pct": 0.05,
    "high_conviction_pct": 0.05,
    "low_conviction_pct": 0.01
  },
  "qlib_models": {
    "NSE500": {"factors": "alpha158_full", "model": "lightgbm_nse500"},
    "NIFTY50": {"factors": "alpha158_large_cap", "model": "lightgbm_nifty50"},
    "NIFTYIT": {"factors": "alpha158_it_sector", "model": "lightgbm_niftyit"},
    "BSE": {"factors": "alpha158_bse", "model": "lightgbm_bse"},
    "BANKING": {"factors": "alpha158_banking", "model": "lightgbm_banking"},
    "PHARMA": {"factors": "alpha158_pharma", "model": "lightgbm_pharma"},
    "default": {"factors": "alpha158_full", "model": "lightgbm_nse500"}
  },
  "kill_switch": false,
  "reasoning": "2-3 sentence explanation of calibration decisions"
}
```

### 3.4 How Components Call the Agent

```python
# Signal Engine calls for weights before scoring
class SignalEngine:
    async def score(self, symbol, market_data):
        weights = await self.calibration_agent.get_signal_weights()
        threshold = await self.calibration_agent.get_signal_threshold()
        
        score = (
            technical_score  * weights["technical"] +
            sentiment_score  * weights["sentiment"] +
            fundamental_score * weights["fundamental"] +
            macro_score      * weights["macro"] +
            candlestick_score * weights["candlestick"] +
            ml_score         * weights["ml_qlib"] +
            debate_conviction * weights["debate_conviction"]
        )
        
        if abs(score) >= threshold:
            return TradeSignal(...)

# Risk Pipeline calls for thresholds before checking
class PreTradeRiskChecker:
    async def check_all(self, order, portfolio):
        thresholds = await self.calibration_agent.get_risk_thresholds()
        
        if order.size_pct > thresholds["max_position_pct"]:
            return RiskCheckResult(passed=False, reason="Position too large")
        # ... all other gates use thresholds dict

# Master Orchestrator calls for agent weights before consensus
class MasterOrchestrator:
    async def compute_consensus(self, signals):
        agent_weights = await self.calibration_agent.get_agent_weights()
        
        weighted_vote = sum(
            signal.strength * signal.action_numeric * agent_weights[signal.agent_name]
            for signal in symbol_signals
        )

# Qlib Pipeline calls for model config per market
class QlibPipeline:
    async def run_factors(self, market="NIFTY50"):
        config = await self.calibration_agent.get_qlib_model_config(market)
        # Uses market-specific factor set and trained model
```

---

## PART 4: LAYER-BY-LAYER DEEP DIVE

### Layer 0: Data Ingestion Engine (MCP Tool-Driven)

*(Same as v2.0 — all DhanHQ v2 tools, Finance Intel DB, news RSS, etc.)*

**CORRECTED Rate Limits:**

| Tool | Rate Limit |
|------|-----------|
| `tool_dhan_quote` / `tool_dhan_ohlc` / `tool_dhan_depth` | 1/sec |
| `tool_dhan_historical` / `tool_dhan_intraday` | 5/sec |
| **`tool_dhan_options_chain`** | **1 per 150 seconds** ← CORRECTED |
| `tool_dhan_expiry_list` | 5/sec |
| `tool_dhan_rolling_options` | 5/sec |
| Order APIs | 10/sec, 7,000/day |

---

### Layer 1: Analyst Agents (Tier 1)

**ALL use Claude Sonnet** (cost-efficient for structured analysis):

- **Technical Analyst:** OHLCV → RSI, MACD, EMA, ADX, BB, ATR, VWAP, OBV, candlestick patterns
- **Sentiment Analyst:** FinBERT + Claude Haiku entity extraction + Finance Intel social narratives
- **Fundamental Analyst:** Screener.in data + Qlib factor scores + earnings
- **Macro/Flow Analyst:** FII/DII, India VIX, RBI calendar, crude, USD/INR

---

### Layer 2: 10 Strategy Sub-Agents (Original v1.0 — PRESERVED)

**All 10 original sub-agents from v1.0 are kept exactly as designed:**

| Agent Name | Specialty | Strategy Type |
|-----------|-----------|--------------|
| `ScalperAgent` | 1-5 min trades | VWAP crossover, volume spike (>2x avg), RSI(14) extremes, EMA(9) vs EMA(21). Top 20 liquid NSE. Min volume 50K/min. Max hold 30 min. |
| `TrendFollowerAgent` | 1-5 day trades | EMA(20) vs EMA(50) crossover, MACD signal cross, ADX > 25 filter, 52-week high breakout scanner. |
| `OptionsAgent` | FnO strategies | IV Crush (sell straddle pre-earnings if IV > 80th pct), Theta Decay (sell OTM covered calls), Momentum (buy ATM calls on breakout), Hedging (buy PE when VIX > 18), Iron Condor construction. Only liquid strikes OI > 5L. |
| `MeanReversionAgent` | Overbought/oversold | Bollinger squeeze→expansion, RSI divergence, Z-score vs 20-day mean (\|z\| > 2), Pairs trading within sector (stat arb). |
| `SentimentAgent` | News-driven trades | Score > +0.6 sustained 2 hrs → BUY, Score < -0.6 sustained 2 hrs → SELL, swing > 0.4 in 30 min → immediate signal. Weight boost by mention_velocity. |
| `FundamentalsAgent` | Value picks | P/E, EPS, ROE analysis from Screener.in. |
| `MacroAgent` | Index/ETF positioning | FII buying 3 consec days → NIFTY BUY, VIX > 22 → reduce + GOLDBEES hedge, VIX < 14 → add equity, USD/INR > 85 → sell IT. |
| `PatternAgent` | Chart patterns | Head & Shoulders, Cup & Handle, double top/bottom, flag/pennant detection. |
| `QuantAgent` | Factor model | Qlib Alpha001-020 + Momentum + Quality + Value + Low Vol. LightGBM predicts 1-day return. Top 10 → BUY, Bottom 10 → SELL. |
| `ETFAgent` | ETF arbitrage | NAV premium/discount arb, sector rotation (macro-based), monthly momentum rebalance, safe haven rotation (equity↔gold↔liquid by VIX). |

**Agent Base Class (preserved from v1.0):**
```python
class BaseAgent:
    # Attributes: name, weight (DYNAMIC via WeightCalibrationAgent), trade_history, performance_metrics
    # Abstract: analyze(market_data, sentiment_data) → AgentSignal
    #           get_entry_price(symbol) → float
    #           get_stop_loss(symbol) → float
    #           get_target(symbol) → float
    # Concrete: update_performance(trade_result) → None
    #           get_weight() → float  # NOW calls WeightCalibrationAgent
    
    async def get_weight(self) -> float:
        """Weight is no longer self-computed — delegates to WeightCalibrationAgent."""
        agent_weights = await self.calibration_agent.get_agent_weights()
        return agent_weights.get(self.name, 0.0)

# AgentSignal dataclass (preserved):
# {symbol, action: BUY/SELL/HOLD, strength: 0.0-1.0,
#  entry, stop_loss, target, position_size_pct, reason, agent_name, timestamp}
```

**Master Orchestrator (preserved from v1.0 + dynamic weights):**
```python
class MasterOrchestrator:
    async def compute_consensus(self, signals):
        agent_weights = await self.calibration_agent.get_agent_weights()
        
        for symbol in universe:
            weighted_vote = sum(
                signal.strength * signal.action_numeric * agent_weights[signal.agent_name]
                for signal in symbol_signals
            )
            threshold = await self.calibration_agent.get_signal_threshold()
            if abs(weighted_vote) >= threshold:
                execute = True
                direction = "BUY" if weighted_vote > 0 else "SELL"
    
    # LLM adjudication prompt (preserved from v1.0) — uses Claude Sonnet
```

### Layer 2B: 22 Backtested Strategies + Candlestick Composites (NEW — Added to agents)

**Source A — Re.Define PDF Strategies:**
- Gap-and-Go (opening range breakout)
- VWAP Reversal (mean reversion to VWAP)
- ORB (Opening Range Breakout 15-min)
- CPR (Central Pivot Range) Support/Resistance

**Source B — Candlestick Composites (backtested on 18.5yr Nifty):**

| Pattern | Profit Factor | Win Rate | Assigned Agent |
|---------|--------------|----------|----------------|
| RSI Divergence + Hammer | 8.51 | 73.7% | MeanReversionAgent |
| EMA Ribbon Cross (8/13/21/34/55) | 6.72 | 77.8% | TrendFollowerAgent |
| Three White Soldiers | 2.90 | 68.9% | PatternAgent |
| Engulfing + Volume Spike | 2.45 | 65.2% | ScalperAgent |
| Morning Star + RSI<30 | 2.12 | 62.1% | MeanReversionAgent |

**Source C — Quantitative Strategies:**
- Qlib Alpha158 factor ranking (top/bottom decile) → QuantAgent
- Statistical arbitrage (pairs within sector) → MeanReversionAgent
- Momentum factor (20-day return rank) → TrendFollowerAgent

**Source D — Options Strategies:**
- IV Crush: Sell straddle pre-earnings if IV > 80th pct → OptionsAgent
- Theta Decay: Sell OTM covered calls → OptionsAgent
- Hedging: Buy PE when VIX > 18 → OptionsAgent
- Max Pain: Position near max pain strike pre-expiry → OptionsAgent

---

### Layer 3-4: Research Agents + Debate Arena

**ALL use Claude Sonnet** (including debate synthesis — changed from Opus per user request):

- **Bull Researcher (Sonnet):** Constructs strongest bullish case from all analyst reports
- **Bear Researcher (Sonnet):** Constructs strongest bearish case or case for NOT trading
- **Risk Researcher (Sonnet):** Independent risk assessment, portfolio-level concerns

**Debate Protocol (Claude Sonnet — NOT Opus):**
```
Round 1: Bull presents thesis → Bear critiques
Round 2: Bear presents counter-thesis → Bull rebuts
Round 3 (optional): If conviction split > 0.3
Risk Researcher provides guardrails throughout

Synthesis: Claude SONNET reads full debate → final direction, conviction, sizing
```

**Portfolio Manager (Claude Sonnet):**
- Approve/Reject/Modify each trade proposal
- Portfolio-level diversification and correlation check
- Capital allocation across approved trades

---

### Layer 6: Signal Engine (★ DYNAMIC Weights)

**The signal engine no longer has hardcoded weights. Every scoring call asks the WeightCalibrationAgent first.**

```python
class DynamicSignalEngine:
    """Signal engine with market-regime-adaptive weights."""
    
    async def score(self, symbol: str, components: dict) -> float:
        # Get regime-adaptive weights from calibration agent
        weights = await self.calibration_agent.get_signal_weights()
        threshold = await self.calibration_agent.get_signal_threshold()
        
        score = sum(
            components.get(key, 0.0) * weight
            for key, weight in weights.items()
        )
        
        return SignalResult(
            score=score,
            threshold=threshold,
            triggered=abs(score) >= threshold,
            regime=weights.get("_regime", "UNKNOWN"),
            weights_used=weights
        )
```

**Example regime adaptations:**

| Regime | Technical | Sentiment | Fundamental | Macro | Candlestick | ML/Qlib | Debate | Threshold |
|--------|-----------|-----------|-------------|-------|-------------|---------|--------|-----------|
| TRENDING | 0.30 | 0.10 | 0.10 | 0.10 | 0.15 | 0.15 | 0.10 | 0.55 |
| MEAN_REVERTING | 0.20 | 0.15 | 0.15 | 0.10 | 0.15 | 0.15 | 0.10 | 0.60 |
| HIGH_VOL | 0.15 | 0.20 | 0.10 | 0.20 | 0.05 | 0.15 | 0.15 | 0.75 |
| LOW_VOL | 0.30 | 0.10 | 0.15 | 0.05 | 0.15 | 0.15 | 0.10 | 0.50 |
| CRISIS | — | — | — | — | — | — | — | ∞ (no trades) |

---

### Layer 7: Risk Management (Original v1.0 — PRESERVED + Dynamic Thresholds)

**Restored exactly from v1.0. The PreTradeRiskChecker, PositionManager, and DrawdownMonitor classes are preserved. Only change: thresholds are now dynamic via WeightCalibrationAgent.**

#### 7A. PreTradeRiskChecker (from v1.0 — preserved)

```python
class PreTradeRiskChecker:
    """PRESERVED from v1.0. All 5 checks remain. Thresholds now dynamic."""
    
    async def check_all(self, order: TradeOrder, portfolio: Portfolio) -> RiskCheckResult:
        thresholds = await self.calibration_agent.get_risk_thresholds()
        
        checks = [
            self.check_position_size(order, portfolio, thresholds),
            self.check_sector_concentration(order, portfolio, thresholds),
            self.check_daily_loss_limit(portfolio, thresholds),
            self.check_market_hours(),
            self.check_news_blackout(order.symbol, thresholds),
            self.check_vix_level(thresholds),
        ]
        return RiskCheckResult(passed=all(checks))
    
    # Max position size: thresholds["max_position_pct"] of capital per stock
    # Max sector exposure: thresholds["max_sector_pct"] of capital
    # Max daily loss limit: thresholds["max_daily_loss_pct"] → auto halt
    # Max drawdown limit: thresholds["max_drawdown_pct"] → pause system, alert
    # No trading thresholds["news_blackout_minutes"] before/after major events
    # VIX > thresholds["vix_defensive_threshold"] → defensive mode
```

#### 7B. PositionManager (from v1.0 — preserved)

```python
class PositionManager:
    """PRESERVED from v1.0. All methods and exit triggers intact."""
    
    # Methods (unchanged):
    #   update_position(trade) → None
    #   get_open_positions() → list[Position]
    #   get_portfolio_delta() → float (for options)
    #   compute_unrealized_pnl() → float
    #   get_position_by_symbol(symbol) → Position
    #   should_exit(position) → tuple[bool, str]
    
    # Exit triggers (unchanged):
    #   - Stop loss hit
    #   - Target achieved
    #   - Trailing stop triggered
    #   - Time-based (intraday squareoff at 15:10 IST)
    #   - Agent signal reversal
    #   - Risk limit breach
    
    # NEW: SL/TP multipliers are now dynamic
    async def compute_stop_loss(self, symbol, entry_price, atr):
        multipliers = await self.calibration_agent.get_sl_tp_multipliers()
        return entry_price - (atr * multipliers["intraday_sl_atr"])
```

#### 7C. DrawdownMonitor (from v1.0 — preserved)

```python
class DrawdownMonitor:
    """PRESERVED from v1.0. Circuit breakers intact. Thresholds now dynamic."""
    
    # Methods (unchanged):
    #   update(portfolio_value) → None
    #   get_current_drawdown() → float
    #   get_max_drawdown() → float
    #   is_circuit_breaker_triggered() → bool
    #   send_alert_if_needed() → None
    
    # Circuit breakers (thresholds now from WeightCalibrationAgent):
    #   Daily loss > thresholds → reduce all new position sizes by 50%
    #   Daily loss > thresholds × 2 → HALT all new positions, only exits allowed
    #   Weekly drawdown > 5% → Pause system, send Telegram alert, require manual restart
    #   Monthly drawdown > 10% → Full system stop
```

#### 7D. Post-Trade Controls (from v1.0 — preserved)

- Dynamic stop-loss: ATR-based (multipliers from WeightCalibrationAgent)
- Trailing stop: multiplier from WeightCalibrationAgent
- Time-based exit: Force close all intraday positions by 3:15 PM IST
- Options Greeks limits: Delta-neutral portfolio maintained within ±0.3

#### 7E. Dhan Native Risk Controls (added in v2.0 — preserved)

- **Kill Switch:** `tool_dhan_kill_switch` — blocks ALL trading for rest of day
- **P&L Exit:** `tool_dhan_pnl_exit` — auto-exit when cumulative P&L hits threshold
- **Exit All:** `tool_dhan_exit_all` — flatten all positions immediately

---

### Layer 8: Qlib Multi-Market Adaptation (★ NEW)

**Instead of one monolithic NSE500 model, NEXUS-II trains separate factor models per market and sector:**

```python
class MultiMarketQlibPipeline:
    """Separate Qlib Alpha158 + LightGBM models per market/sector."""
    
    MARKET_CONFIGS = {
        "NSE500": {
            "universe": "all NSE500 stocks",
            "factors": "alpha158_full",  # Full 158 factors
            "features": ["momentum_5d", "momentum_20d", "reversal_1d",
                         "volume_momentum", "turnover_rate", 
                         "realized_vol_20d", "vol_of_vol",
                         "pe_ratio", "pb_ratio", "roe"],
            "target": "5d_return_rank",
            "retrain": "weekly_sunday",
        },
        "NIFTY50": {
            "universe": "Nifty 50 constituents only",
            "factors": "alpha158_large_cap",  # Focus on large-cap factors
            "features": ["momentum_20d", "fii_flow_5d", "sector_rotation",
                         "earnings_surprise", "institutional_ownership"],
            "target": "1d_return",
            "retrain": "weekly_sunday",
        },
        "NIFTYIT": {
            "universe": "Nifty IT index stocks (TCS, INFY, WIPRO, HCLTECH, etc.)",
            "factors": "alpha158_it_sector",  # IT-specific factors
            "features": ["usd_inr_momentum", "us_tech_correlation",
                         "deal_win_sentiment", "attrition_proxy",
                         "revenue_growth_qoq", "margin_expansion"],
            "target": "5d_return_rank",
            "retrain": "weekly_sunday",
        },
        "BSE": {
            "universe": "BSE 500 (for stocks not in NSE)",
            "factors": "alpha158_bse",
            "features": ["momentum_5d", "volume_momentum", "pe_ratio"],
            "target": "5d_return_rank",
            "retrain": "bi_weekly",
        },
        "BANKING": {
            "universe": "NIFTYBANK + PSU banks",
            "factors": "alpha158_banking",
            "features": ["npa_ratio_change", "casa_ratio", "credit_growth",
                         "nim_expansion", "rbi_rate_sensitivity",
                         "fii_banking_flow"],
            "target": "1d_return",
            "retrain": "weekly_sunday",
        },
        "PHARMA": {
            "universe": "NIFTYPHARMA stocks",
            "factors": "alpha158_pharma",
            "features": ["fda_approval_sentiment", "us_revenue_pct",
                         "r_and_d_intensity", "patent_cliff_proximity",
                         "api_price_trends"],
            "target": "5d_return_rank",
            "retrain": "weekly_sunday",
        },
    }
    
    async def run_daily_factors(self):
        """Run factor computation for all configured markets."""
        for market, config in self.MARKET_CONFIGS.items():
            # Get which model to use from WeightCalibrationAgent
            model_config = await self.calibration_agent.get_qlib_model_config(market)
            
            factors = self.compute_factors(market, model_config["factors"])
            predictions = self.predict(factors, model_config["model"])
            self.store_rankings(market, predictions)
    
    def get_top_picks(self, market: str, n: int = 10) -> list:
        """Get top N stocks by predicted return for a given market."""
        return self.rankings[market][:n]
    
    def get_bottom_picks(self, market: str, n: int = 10) -> list:
        """Get bottom N stocks (sell/short candidates)."""
        return self.rankings[market][-n:]
```

---

## PART 5: DEPLOYMENT ARCHITECTURE (★ NEW — GitHub + Supabase + Cloudflare)

### 5.1 Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                    DEPLOYMENT STACK                        │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  GitHub (Source & CI/CD)                                   │
│  ├── nexus-ii/ repo                                       │
│  ├── GitHub Actions: test → build → deploy                │
│  └── Branch protection: main (prod), dev (staging)        │
│                                                           │
│  Supabase (Database + Auth + Realtime)                    │
│  ├── PostgreSQL: trades, positions, signals, agents,      │
│  │   sentiment_cache, factor_scores, audit_log            │
│  ├── Auth: API key management, dashboard login            │
│  ├── Realtime: live position updates → dashboard          │
│  ├── Edge Functions: lightweight webhooks (Dhan postback) │
│  └── Storage: model weights, backtest reports             │
│                                                           │
│  Cloudflare Workers (Trading Bot Runtime)                 │
│  ├── Main bot process (Python via Cloudflare Workers)     │
│  ├── Cron Triggers: scheduler replacement                 │
│  ├── KV Store: cache (replace Redis)                      │
│  ├── R2: large file storage (historical data)             │
│  ├── Durable Objects: stateful position tracking          │
│  └── Static IP: via Cloudflare Tunnel for Dhan API        │
│                                                           │
│  Vercel (Dashboard Frontend)                              │
│  ├── Next.js Streamlit-alternative dashboard              │
│  ├── Real-time portfolio view via Supabase Realtime       │
│  ├── Agent leaderboard, sentiment heatmap                 │
│  └── Git-push deploy from GitHub                          │
│                                                           │
│  Telegram (Alerts + Approval)                             │
│  └── Bot: trade proposals, PnL, risk alerts               │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

### 5.2 Supabase Schema (replaces SQLite)

```sql
-- Core trading tables
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
    status TEXT DEFAULT 'OPEN',
    opened_at TIMESTAMPTZ DEFAULT NOW(),
    closed_at TIMESTAMPTZ,
    metadata JSONB
);

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

CREATE TABLE calibration_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    regime TEXT,
    signal_weights JSONB,
    risk_thresholds JSONB,
    agent_weights JSONB,
    reasoning TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE sentiment_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol TEXT NOT NULL,
    score DECIMAL,
    source TEXT,
    tier INT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

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

-- Enable Realtime for dashboard
ALTER PUBLICATION supabase_realtime ADD TABLE trades;
ALTER PUBLICATION supabase_realtime ADD TABLE agent_performance;
```

### 5.3 Cloudflare Workers Configuration

```toml
# wrangler.toml
name = "nexus-ii-bot"
main = "src/worker.py"
compatibility_date = "2026-04-01"

[vars]
TRADING_MODE = "PAPER_TRADE"
SUPABASE_URL = "https://your-project.supabase.co"

# Cron triggers (replaces APScheduler)
[triggers]
crons = [
    "0 8 * * 1-5",    # 08:00 IST pre-market
    "30 8 * * 1-5",   # 08:30 macro analysis
    "45 8 * * 1-5",   # 08:45 factor scores
    "0 9 * * 1-5",    # 09:00 morning brief
    "*/15 9-15 * * 1-5", # Every 15 min during market
    "10 15 * * 1-5",  # 15:10 squareoff
    "0 16 * * 1-5",   # 16:00 daily summary
    "0 22 * * 0",     # Sunday 22:00 retrain
]

# KV namespace (replaces Redis)
[[kv_namespaces]]
binding = "CACHE"
id = "your-kv-id"

# R2 bucket (historical data)
[[r2_buckets]]
binding = "STORAGE"
bucket_name = "nexus-data"

# Secrets (set via wrangler secret)
# DHAN_ACCESS_TOKEN, CLAUDE_API_KEY, TELEGRAM_BOT_TOKEN, SUPABASE_ANON_KEY
```

### 5.4 GitHub Actions CI/CD

```yaml
# .github/workflows/deploy.yml
name: Deploy NEXUS-II
on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v --cov

  deploy-bot:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: cloudflare/wrangler-action@v3
        with:
          apiToken: ${{ secrets.CF_API_TOKEN }}
          command: deploy

  deploy-dashboard:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: amondnet/vercel-action@v25
        with:
          vercel-token: ${{ secrets.VERCEL_TOKEN }}
          working-directory: ./dashboard
```

---

## PART 6: PROJECT STRUCTURE

```
nexus-ii/
├── .github/
│   └── workflows/
│       └── deploy.yml              # CI/CD: test → Cloudflare + Vercel
├── core/
│   ├── mcp_tools/
│   │   ├── dhan_tools.py           # All DhanHQ v2 API wrappers
│   │   ├── data_tools.py           # News, macro, sentiment tools
│   │   ├── compute_tools.py        # Technical indicators, factors
│   │   └── tool_registry.py        # MCP tool registration
│   ├── calibration/
│   │   ├── weight_calibration_agent.py  # ★ The central weight brain
│   │   ├── regime_detector.py      # Market regime classification
│   │   └── safety_bounds.py        # Hard limits LLM can't override
│   ├── agents/
│   │   ├── base_agent.py           # Abstract agent class
│   │   ├── scalper_agent.py        # (v1.0 preserved)
│   │   ├── trend_agent.py          # (v1.0 preserved)
│   │   ├── options_agent.py        # (v1.0 preserved)
│   │   ├── mean_reversion_agent.py # (v1.0 preserved)
│   │   ├── sentiment_agent.py      # (v1.0 preserved)
│   │   ├── fundamentals_agent.py   # (v1.0 preserved)
│   │   ├── macro_agent.py          # (v1.0 preserved)
│   │   ├── pattern_agent.py        # (v1.0 preserved)
│   │   ├── quant_agent.py          # (v1.0 preserved)
│   │   ├── etf_agent.py            # (v1.0 preserved)
│   │   ├── master_orchestrator.py  # (v1.0 preserved + dynamic weights)
│   │   ├── analysts/               # (v2.0 TradingAgents tier 1)
│   │   │   ├── technical_analyst.py
│   │   │   ├── sentiment_analyst.py
│   │   │   ├── fundamental_analyst.py
│   │   │   └── macro_analyst.py
│   │   ├── researchers/            # (v2.0 TradingAgents tier 2)
│   │   │   ├── bull_researcher.py
│   │   │   ├── bear_researcher.py
│   │   │   └── risk_researcher.py
│   │   ├── debate/
│   │   │   └── debate_arena.py     # Claude Sonnet debate
│   │   └── portfolio_manager.py    # Claude Sonnet PM
│   ├── signal_engine/
│   │   ├── dynamic_signal_scorer.py # ★ Dynamic weights
│   │   ├── strategy_library.py     # 22 backtested strategies
│   │   └── candlestick_patterns.py # Pattern detection
│   ├── sentiment/
│   │   ├── finbert_engine.py
│   │   ├── llm_enricher.py         # Claude Haiku extraction
│   │   ├── sector_aggregator.py
│   │   └── sentiment_store.py
│   ├── brain/
│   │   ├── multi_market_qlib.py    # ★ Multi-market factor models
│   │   ├── model_registry.py
│   │   └── finrl_trainer.py        # Optional (Month 3+)
│   ├── risk/                       # (v1.0 preserved)
│   │   ├── pre_trade_checks.py     # PreTradeRiskChecker (dynamic thresholds)
│   │   ├── position_manager.py     # PositionManager (dynamic SL/TP)
│   │   └── drawdown_monitor.py     # DrawdownMonitor (dynamic circuit breakers)
│   ├── execution/
│   │   ├── dhan_executor.py
│   │   ├── paper_trader.py
│   │   └── order_manager.py
│   ├── monitoring/
│   │   ├── telegram_bot.py         # Alerts + approval flow
│   │   ├── guard_watchdog.py
│   │   └── report_generator.py
│   └── mirofish/
│       ├── scenario_builder.py
│       ├── simulation_runner.py
│       └── report_extractor.py
├── dashboard/                      # Vercel-deployed Next.js frontend
│   ├── package.json
│   ├── app/
│   │   ├── page.tsx                # Portfolio overview
│   │   ├── agents/page.tsx         # Agent leaderboard
│   │   ├── signals/page.tsx        # Signal history
│   │   └── calibration/page.tsx    # Weight calibration log
│   └── lib/
│       └── supabase.ts             # Supabase client
├── skills/
│   └── weight-calibration/
│       └── SKILL.md                # Weight Calibration Skill definition
├── config/
│   ├── settings.py
│   ├── universe.py
│   └── strategy_params.py
├── tests/
├── supabase/
│   └── migrations/                 # SQL migration files
├── main.py
├── wrangler.toml                   # Cloudflare Workers config
├── .env.example
└── requirements.txt
```

---

## PART 7: TECHNOLOGY STACK

| Layer | Technology | Why |
|-------|-----------|-----|
| Language | Python 3.11+ | Ecosystem, async support |
| Broker API | DhanHQ v2 REST + WebSocket | Indian market native, Super Orders, Kill Switch |
| LLM (ALL tiers) | **Claude Sonnet 4.6** | Sonnet for everything: debate, PM, analysts, calibration |
| LLM (Entity Extraction) | Claude Haiku 4.5 | Fastest, cheapest for high-throughput NLP |
| Quant Research | Qlib Alpha158 + LightGBM (multi-market) | Per-market factor models |
| Sentiment NLP | FinBERT (ProsusAI) | Financial sentiment classifier |
| Swarm Simulation | MiroFish (weekend) | Emergent consensus |
| Backtesting | vectorbt | Fast vectorized |
| **Database** | **Supabase (PostgreSQL)** | Replaces SQLite; realtime, auth, global |
| **Cache** | **Cloudflare KV** | Replaces Redis; edge-cached globally |
| **Object Storage** | **Cloudflare R2** | Historical data, model weights; zero egress |
| **Bot Runtime** | **Cloudflare Workers** | Serverless, cron triggers, global edge |
| **Dashboard** | **Vercel + Next.js** | Git-push deploy, SSR, Supabase Realtime |
| **Source Control** | **GitHub + Actions** | CI/CD, branch protection, automated deploy |
| Notifications | python-telegram-bot | Alerts + approval flow |
| Logging | structlog (JSON) → Supabase | Structured, queryable |
| Config | Pydantic BaseSettings + env vars | Type-safe |
| Testing | pytest + pytest-asyncio | Standard |

---

## PART 8: LLM USAGE (ALL Sonnet)

| Component | Model | Calls/Day (est.) | Cost/Day (est.) |
|-----------|-------|-------------------|-----------------|
| Weight Calibration Agent | Sonnet 4.6 | ~40 (every 15 min × 6.5 hrs) | ~$0.80 |
| Technical Analyst | Sonnet 4.6 | ~26 (every 15 min) | ~$0.50 |
| Sentiment Analyst | Haiku 4.5 | ~200 (entity extraction) | ~$0.20 |
| Fundamental Analyst | Sonnet 4.6 | ~5 (EOD) | ~$0.10 |
| Macro Analyst | Sonnet 4.6 | ~10 | ~$0.20 |
| Bull/Bear Researchers | Sonnet 4.6 | ~26 each | ~$1.00 |
| Debate Arena | **Sonnet 4.6** | ~26 | ~$0.50 |
| Portfolio Manager | **Sonnet 4.6** | ~26 | ~$0.50 |
| Master Orchestrator (v1 adjudication) | Sonnet 4.6 | ~10 | ~$0.20 |
| **Total** | | ~400 | **~$4.00/day** |

---

## PART 9: CRITICAL RULES

```
1.  ALWAYS start in PAPER_TRADE mode. Never skip this.
2.  NEVER hardcode API keys — use env vars + Cloudflare Secrets.
3.  EVERY order must pass ALL risk gates (dynamic thresholds) before execution.
4.  ALL time logic must use IST timezone (zoneinfo.ZoneInfo("Asia/Kolkata")).
5.  Log EVERYTHING — structlog JSON → Supabase.
6.  Use async/await for all network calls.
7.  Handle ALL exceptions gracefully — system must never crash silently.
8.  Every module must have unit tests.
9.  Type hints, docstrings, no magic numbers.
10. Capital preservation > profit. When in doubt, HOLD.
11. Dhan rate limits: 10 orders/sec, 5 data/sec, 1 option chain/150sec.
12. Set Dhan static IP via Cloudflare Tunnel before live orders.
13. Use Super Orders as default order type.
14. MiroFish simulation only on weekends.
15. Claude Sonnet for ALL tiers (debate, PM, analysts, calibration). Haiku for extraction only.
16. All LLM calls: timeout 30s, retry 3x exponential backoff.
17. Never trade ±30 min around RBI policy, Budget, scheduled earnings.
18. Dhan Kill Switch is last resort — blocks ALL trading for the day.
19. WeightCalibrationAgent MUST be called before ANY weight is used. Never hardcode.
20. Safety bounds on WeightCalibrationAgent output are HARD — LLM cannot override.
21. Deploy via GitHub Actions → Cloudflare Workers (bot) + Vercel (dashboard).
22. All data persists in Supabase PostgreSQL. No local SQLite in production.
```

---

## PART 10: BUILD ORDER

```
Phase 1 — Foundation (Week 1):
  1. GitHub repo setup + Supabase project + Cloudflare account
  2. config/ (settings.py, universe.py, strategy_params.py)
  3. supabase/migrations/ (all tables)
  4. core/mcp_tools/dhan_tools.py (all DhanHQ v2 wrappers)
  5. core/mcp_tools/data_tools.py (news, VIX, FII/DII)
  6. core/calibration/weight_calibration_agent.py ★
  7. core/execution/paper_trader.py
  8. core/risk/ (PreTradeRiskChecker + PositionManager + DrawdownMonitor — v1.0 code + dynamic thresholds)
  9. core/monitoring/telegram_bot.py
  10. main.py + wrangler.toml (Cloudflare cron triggers)
  11. Tests for Phase 1

Phase 2 — All 10 Agents + Signal Engine (Week 2):
  12. core/agents/base_agent.py (with dynamic weight from calibration agent)
  13. All 10 sub-agents (scalper, trend, options, mean_reversion, sentiment, fundamentals, macro, pattern, quant, etf)
  14. core/agents/master_orchestrator.py (v1.0 + dynamic weights)
  15. core/signal_engine/dynamic_signal_scorer.py
  16. core/signal_engine/strategy_library.py (22 strategies)
  17. core/signal_engine/candlestick_patterns.py
  18. core/sentiment/finbert_engine.py + llm_enricher.py
  19. Tests for Phase 2

Phase 3 — TradingAgents Debate Layer (Week 3):
  20. core/agents/analysts/ (4 analyst agents)
  21. core/agents/researchers/ (bull, bear, risk)
  22. core/agents/debate/debate_arena.py (Claude Sonnet)
  23. core/agents/portfolio_manager.py (Claude Sonnet)
  24. Telegram approval flow (approve/reject/modify)
  25. Tests for Phase 3

Phase 4 — ML Brain + Dashboard (Week 4):
  26. core/brain/multi_market_qlib.py (NSE500, NIFTY50, NIFTYIT, BSE, Banking, Pharma)
  27. Finance Intelligence DB setup
  28. dashboard/ (Next.js + Supabase Realtime on Vercel)
  29. core/monitoring/report_generator.py
  30. GitHub Actions CI/CD pipeline
  31. Tests for Phase 4

Phase 5 — Paper Trading (Month 2):
  32. Full integration on Dhan paper trading mode
  33. 30-day forward test, ₹50K virtual capital
  34. Cloudflare Workers production deployment
  35. Guard watchdog
  36. Performance gate: WR ≥ 55%, Sharpe ≥ 0.5, MaxDD ≤ 8%, PF ≥ 1.5

Phase 6 — Live (Month 3+):
  37. TRADING_MODE = LIVE
  38. Cloudflare Tunnel for Dhan static IP
  39. Dhan P&L Exit + Kill Switch configured
  40. Start with ₹50,000 real capital
  41. Optional: FinRL training on live data
  42. Optional: MiroFish weekend simulation
```

---

## PART 11: TIMELINE & CAPITAL

| Phase | Duration | Goal | Capital Risk |
|-------|----------|------|-------------|
| Phase 1 | Week 1 | Foundation + Risk + Calibration Agent | ₹0 |
| Phase 2 | Week 2 | All 10 Agents + Dynamic Signal Engine | ₹0 |
| Phase 3 | Week 3 | Debate Layer + Dashboard | ₹0 |
| Phase 4 | Week 4 | Multi-Market ML + Reports | ₹0 |
| Phase 5 | Month 2 | 30-day paper trading on Dhan | ₹0 (virtual) |
| Phase 6 | Month 3+ | Live with real capital | ₹50,000 max |

**Target (Month 6):** WR 65-70%, Sharpe 1.5-2.0, Monthly return 3-5%

---

## PART 12: WEIGHT CALIBRATION SKILL DEFINITION

**Save as `/skills/weight-calibration/SKILL.md`:**

```markdown
---
name: weight-calibration
description: >
  Dynamic weight calibration skill for the NEXUS-II trading system. 
  Called by EVERY component before using any weight, threshold, or parameter.
  Reads current market conditions (VIX, regime, drawdown, agent performance,
  sector momentum) and returns calibrated values for signal engine weights,
  risk gate thresholds, agent weights, SL/TP multipliers, position sizing,
  and Qlib model selection. Uses Claude Sonnet with 15-min cache TTL.
  
  ALWAYS use this skill when:
  - Any component needs signal engine weights
  - Any component needs risk thresholds
  - Any component needs agent vote weights
  - Any component needs SL/TP parameters
  - Any component needs position sizing
  - Any component needs to select a Qlib model for a market/sector
  - The market regime changes
  - At the start of each trading day
  - After any circuit breaker event

triggers:
  - "what weights should I use"
  - "calibrate weights"
  - "market regime changed"
  - "recalibrate"
  - "get risk thresholds"
  - "get signal weights"
  - "get agent weights"
---

# Weight Calibration Skill

## Purpose
Centralized dynamic weight management for ALL components in NEXUS-II.
No component ever uses a hardcoded weight — all weights flow through this skill.

## Interface
- Input: Current market state (VIX, regime, drawdown, agent Sharpe scores, etc.)
- Output: Complete calibration JSON with all weights and thresholds
- Cache: 15-min TTL during market hours, 1-hour off-hours
- Safety: Hard bounds enforced AFTER LLM output — LLM cannot exceed safety limits

## Safety Bounds (NON-NEGOTIABLE)
- max_position_pct: NEVER > 10%
- max_sector_pct: NEVER > 35%
- max_daily_loss_pct: NEVER > 3%
- max_drawdown_pct: NEVER > 12%
- signal_threshold: ALWAYS between 0.40 and 0.80
- signal_weights: MUST sum to 1.0
- agent_weights: MUST sum to 1.0
- kill_switch: true ONLY when VIX > 28 or drawdown > 10%

## Files
- weight_calibration_agent.py — Main agent class
- regime_detector.py — Market regime classification (TRENDING/MEAN_REVERTING/HIGH_VOL/LOW_VOL/CRISIS)
- safety_bounds.py — Hard limits enforcement
```

---

**⚠️ DISCLAIMER:** This system is for educational and research purposes. Indian securities trading involves significant financial risk. Past performance of algorithms does not guarantee future results. SEBI regulations apply to algorithmic trading. Consult SEBI guidelines and a registered investment advisor before live deployment. This document does not constitute financial advice.

---

*Document generated for: Sourav Saha Roy | Hyderabad, India | April 2026*
*System Architecture Version: 2.1*
*Key innovations: Dynamic WeightCalibrationAgent, Multi-Market Qlib, GitHub+Supabase+Cloudflare deployment*
