"""
NEXUS-II — Weight Calibration Agent
Central brain for ALL dynamic weights, thresholds, and risk parameters.

Every component in the system calls this agent before using any weight.
Results are cached for 15 minutes during market hours (1 hour otherwise)
to avoid redundant LLM calls across concurrent agent coroutines.

Flow:
    Component.method()
        → calibration_agent.get_<param>()
            → _get_calibration()   (checks TTL cache)
                → _gather_market_state()
                → _call_claude(market_state)   [on cache miss]
                → safety_bounds.enforce(raw)
                → cache result
            ← CalibrationResult
        ← specific param slice
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from zoneinfo import ZoneInfo

import anthropic

from config.settings import CALIBRATION_TTL_MARKET_HOURS, CALIBRATION_TTL_OFF_HOURS, LLM_MAIN
from core.calibration import safety_bounds
from core.calibration.regime_detector import Regime, RegimeDetector, RegimeResult

log = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")
_MARKET_OPEN_H  = 9
_MARKET_OPEN_M  = 15
_MARKET_CLOSE_H = 15
_MARKET_CLOSE_M = 30

# ── System prompt ─────────────────────────────────────────────────────────────
WEIGHT_CALIBRATION_SYSTEM_PROMPT = """\
You are the Weight Calibration Agent for NEXUS-II, an AI-powered automated \
trading system for Indian markets (NSE/BSE).

Your job: Given the current market state JSON, output calibrated weights and \
thresholds that maximise risk-adjusted returns while preserving capital.

== MARKET REGIMES ==
TRENDING: ADX > 25, clear directional move, low mean-reversion
  → Boost: technical, trend_follower, quant, candlestick
  → Reduce: mean_reversion, scalper
  → signal_threshold: 0.55–0.65

MEAN_REVERTING: ADX < 20, range-bound, high mean-reversion
  → Boost: mean_reversion, sentiment, fundamental
  → Reduce: trend_follower, macro
  → signal_threshold: 0.60

HIGH_VOL: VIX > 20, large intraday ranges, fear regime
  → Boost: risk gates (tighten ALL), options (hedging)
  → Halve all position sizes
  → Reduce: scalper (too risky), new longs
  → signal_threshold: 0.70–0.80  (only high-conviction trades)
  → intraday_sl_atr: 3.0+ (wide SL to avoid noise)

LOW_VOL: VIX < 14, tight ranges, complacency
  → Boost: scalper, theta-decay options
  → Increase position sizes moderately
  → Reduce: macro, hedging
  → signal_threshold: 0.50–0.55  (more trades OK)
  → intraday_sl_atr: 1.5 (tight SL)

CRISIS: VIX > 28, circuit-breaker territory
  → ALL new trades halted, exits only
  → Return kill_switch: true
  → All position sizes → 0

== AGENT PERFORMANCE RULES ==
- Agent 30-day Sharpe < 0  → weight = 0.0  (muted — losing strategy)
- Agent 30-day Sharpe > 2  → weight = max(normalised) among all agents
- Agent < 30 days history  → weight = 1/N  (equal weight, no track record)
- After muting/boosting, normalise all agent_weights to sum = 1.0

== FII/DII INTERPRETATION ==
- FII net buying > ₹2000cr (3 consecutive days) → boost macro, trend signals
- FII net selling > ₹2000cr → tighten risk gates, boost sentiment weight
- DII offsetting FII sell → moderate impact (DII provides floor)

== TIME-OF-DAY ADJUSTMENTS ==
- 09:15–09:45 IST (opening volatility): tighten signal_threshold +0.05
- 14:45–15:30 IST (expiry/closing volatility): reduce position_sizing.default_pct −0.5%
- 10:00–14:00 IST (stable zone): use regime defaults

== OUTPUT FORMAT ==
Return ONLY valid JSON — no markdown, no explanation outside the JSON:
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
    "scalper": 0.10,
    "trend_follower": 0.12,
    "options": 0.10,
    "mean_reversion": 0.10,
    "sentiment": 0.10,
    "fundamentals": 0.10,
    "macro": 0.10,
    "pattern": 0.08,
    "quant": 0.12,
    "etf": 0.08
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
    "NSE500":  {"factors": "alpha158_full",       "model": "lightgbm_nse500"},
    "NIFTY50": {"factors": "alpha158_large_cap",  "model": "lightgbm_nifty50"},
    "NIFTYIT": {"factors": "alpha158_it_sector",  "model": "lightgbm_niftyit"},
    "BSE":     {"factors": "alpha158_bse",        "model": "lightgbm_bse"},
    "BANKING": {"factors": "alpha158_banking",    "model": "lightgbm_banking"},
    "PHARMA":  {"factors": "alpha158_pharma",     "model": "lightgbm_pharma"},
    "default": {"factors": "alpha158_full",       "model": "lightgbm_nse500"}
  },
  "kill_switch": false,
  "reasoning": "2-3 sentence explanation of the calibration decisions."
}
"""


# ── Calibration result dataclass ──────────────────────────────────────────────

@dataclass
class CalibrationResult:
    """Typed wrapper around the raw calibration dict."""
    raw: dict
    regime: Regime
    cached_at: float  # unix timestamp
    reasoning: str

    # ── Convenience accessors ──────────────────────────────────────────────
    @property
    def signal_weights(self) -> dict[str, float]:
        return self.raw["signal_weights"]

    @property
    def signal_threshold(self) -> float:
        return self.raw["signal_threshold"]

    @property
    def agent_weights(self) -> dict[str, float]:
        return self.raw["agent_weights"]

    @property
    def risk_thresholds(self) -> dict[str, Any]:
        return self.raw["risk_thresholds"]

    @property
    def sl_tp_multipliers(self) -> dict[str, float]:
        return self.raw["sl_tp_multipliers"]

    @property
    def position_sizing(self) -> dict[str, float]:
        return self.raw["position_sizing"]

    @property
    def kill_switch(self) -> bool:
        return self.raw.get("kill_switch", False)

    def qlib_model_config(self, market: str) -> dict:
        models = self.raw.get("qlib_models", {})
        return models.get(market, models.get("default", {}))


# ── Main agent class ──────────────────────────────────────────────────────────

class WeightCalibrationAgent:
    """
    Central brain for ALL dynamic weights in NEXUS-II.

    Called by every component before using any weight, threshold, or risk
    parameter. Uses Claude Sonnet to reason about market conditions.
    Results are cached (TTL-based) to avoid redundant LLM calls.

    Usage:
        agent = WeightCalibrationAgent(claude_client, market_data_tools)

        weights   = await agent.get_signal_weights()
        threshold = await agent.get_signal_threshold()
        risk      = await agent.get_risk_thresholds()
    """

    def __init__(
        self,
        claude_client: anthropic.AsyncAnthropic,
        market_data_tools: Any,
        *,
        ttl_market_hours: int = CALIBRATION_TTL_MARKET_HOURS,
        ttl_off_hours: int = CALIBRATION_TTL_OFF_HOURS,
    ) -> None:
        self._client = claude_client
        self._tools = market_data_tools
        self._ttl_market = ttl_market_hours
        self._ttl_off    = ttl_off_hours

        self._cache: Optional[CalibrationResult] = None
        self._lock = asyncio.Lock()     # prevents stampede on cache miss
        self._regime_detector = RegimeDetector()

    # ── Public accessors (called by all other components) ─────────────────

    async def get_signal_weights(self) -> dict[str, float]:
        """Returns 7-component signal engine weights (sum to 1.0)."""
        return (await self._get_calibration()).signal_weights

    async def get_signal_threshold(self) -> float:
        """Returns dynamic consensus threshold (0.40–0.80)."""
        return (await self._get_calibration()).signal_threshold

    async def get_agent_weights(self) -> dict[str, float]:
        """Returns vote weight for each of the 10 strategy sub-agents."""
        return (await self._get_calibration()).agent_weights

    async def get_risk_thresholds(self) -> dict[str, Any]:
        """Returns dynamic risk gate thresholds."""
        return (await self._get_calibration()).risk_thresholds

    async def get_sl_tp_multipliers(self) -> dict[str, float]:
        """Returns ATR multipliers for stop-loss and take-profit."""
        return (await self._get_calibration()).sl_tp_multipliers

    async def get_position_sizing(self) -> dict[str, float]:
        """Returns position sizing parameters."""
        return (await self._get_calibration()).position_sizing

    async def get_qlib_model_config(self, market: str) -> dict:
        """Returns which Qlib factor model to use for a given market/sector."""
        return (await self._get_calibration()).qlib_model_config(market)

    async def is_kill_switch_active(self) -> bool:
        """Returns True if the LLM determined a CRISIS regime (VIX > 28)."""
        return (await self._get_calibration()).kill_switch

    async def kill_switch_active(self) -> bool:
        """Alias for is_kill_switch_active()."""
        return await self.is_kill_switch_active()

    async def calibrate(self) -> dict:
        """Refresh calibration via _call_claude and return raw dict. Caches result."""
        async with self._lock:
            now = time.monotonic()
            if self._cache is not None:
                ttl = (
                    getattr(self, '_ttl_market', CALIBRATION_TTL_MARKET_HOURS)
                    if self._is_market_hours()
                    else getattr(self, '_ttl_off', CALIBRATION_TTL_OFF_HOURS)
                )
                age = now - self._cache.cached_at
                if age < ttl:
                    return self._cache.raw

            raw_json = await self._call_claude({})
            raw = json.loads(raw_json) if isinstance(raw_json, str) else raw_json
            raw = safety_bounds.enforce(raw)

            regime_str = raw.get("regime", raw.get("market_regime", "UNKNOWN"))
            try:
                regime = Regime(regime_str)
            except ValueError:
                regime = Regime.UNKNOWN

            self._cache = CalibrationResult(
                raw=raw,
                regime=regime,
                cached_at=time.monotonic(),
                reasoning=raw.get("reasoning", ""),
            )
            return raw

    async def get_current_regime(self) -> Regime:
        """Returns the current market regime label."""
        return (await self._get_calibration()).regime

    async def get_full_calibration(self) -> CalibrationResult:
        """Returns the full calibration result (for logging / dashboard)."""
        return await self._get_calibration()

    # ── Cache logic ───────────────────────────────────────────────────────

    async def _get_calibration(self) -> CalibrationResult:
        """
        Returns cached calibration if fresh, otherwise triggers a new LLM call.
        Uses an asyncio.Lock to prevent simultaneous LLM calls (stampede protection).
        """
        now = time.monotonic()

        # Fast path: cache hit (no lock needed for read)
        if self._cache is not None:
            ttl = (
                getattr(self, '_ttl_market', CALIBRATION_TTL_MARKET_HOURS)
                if self._is_market_hours()
                else getattr(self, '_ttl_off', CALIBRATION_TTL_OFF_HOURS)
            )
            age = now - self._cache.cached_at
            if age < ttl:
                log.debug(
                    "CalibrationAgent: cache hit (age=%.0fs, ttl=%ds, regime=%s)",
                    age, ttl, self._cache.regime.value,
                )
                return self._cache

        # Slow path: cache miss — acquire lock so only one coroutine calls LLM
        async with self._lock:
            # Double-check after acquiring lock (another coroutine may have refreshed)
            if self._cache is not None:
                ttl = (
                    getattr(self, '_ttl_market', CALIBRATION_TTL_MARKET_HOURS)
                    if self._is_market_hours()
                    else getattr(self, '_ttl_off', CALIBRATION_TTL_OFF_HOURS)
                )
                if (now - self._cache.cached_at) < ttl:
                    return self._cache

            log.info("CalibrationAgent: cache miss — refreshing via LLM")
            return await self._refresh_cache()

    async def _refresh_cache(self) -> CalibrationResult:
        """Gather market state, call LLM, enforce safety bounds, cache result."""
        try:
            market_state = await self._gather_market_state()
            raw = await self._call_claude(market_state)
            raw = safety_bounds.enforce(raw)

            regime_str = raw.get("market_regime", "UNKNOWN")
            try:
                regime = Regime(regime_str)
            except ValueError:
                log.warning("CalibrationAgent: unknown regime '%s', defaulting to UNKNOWN", regime_str)
                regime = Regime.UNKNOWN

            result = CalibrationResult(
                raw=raw,
                regime=regime,
                cached_at=time.monotonic(),
                reasoning=raw.get("reasoning", ""),
            )
            self._cache = result
            log.info(
                "CalibrationAgent: refreshed — regime=%s threshold=%.2f reasoning='%s'",
                regime.value,
                raw.get("signal_threshold", 0.60),
                raw.get("reasoning", "")[:120],
            )
            return result

        except Exception as exc:
            log.error("CalibrationAgent: LLM refresh failed — %s", exc, exc_info=True)
            if self._cache is not None:
                log.warning("CalibrationAgent: using stale cache due to error")
                return self._cache
            log.warning("CalibrationAgent: no cache available — using static defaults")
            return self._build_default_result()

    # ── LLM call ──────────────────────────────────────────────────────────

    async def _call_claude(self, market_state: dict) -> dict:
        """Call Claude Sonnet with the market state and return parsed JSON."""
        prompt = self._build_prompt(market_state)

        response = await self._client.messages.create(
            model=LLM_MAIN,
            max_tokens=2048,
            system=WEIGHT_CALIBRATION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        raw_text = response.content[0].text.strip()

        # Strip any accidental markdown fences
        if raw_text.startswith("```"):
            lines = raw_text.splitlines()
            raw_text = "\n".join(
                line for line in lines if not line.startswith("```")
            ).strip()

        try:
            return json.loads(raw_text)
        except json.JSONDecodeError as exc:
            log.error(
                "CalibrationAgent: JSON parse error — %s\nRaw response:\n%s",
                exc, raw_text[:500],
            )
            raise

    def _build_prompt(self, market_state: dict) -> str:
        return (
            "Current market state (IST):\n"
            + json.dumps(market_state, indent=2, default=str)
            + "\n\nOutput the calibrated JSON now."
        )

    # ── Market state collection ───────────────────────────────────────────

    async def _gather_market_state(self) -> dict:
        """
        Collect all indicators the LLM needs. Runs async data fetches
        concurrently where possible.
        """
        now_ist = datetime.now(IST)

        # Concurrent async fetches
        (
            india_vix,
            nifty_change_pct,
            fii_dii_flow,
            sector_momentum,
            market_breadth,
            options_iv_percentile,
        ) = await asyncio.gather(
            self._tools.get_india_vix(),
            self._tools.get_nifty_change(),
            self._tools.get_fii_dii(),
            self._tools.get_sector_momentum(),
            self._tools.get_advance_decline_ratio(),
            self._tools.get_avg_iv_percentile(),
            return_exceptions=True,
        )

        # Replace exceptions with sensible defaults (don't let one failing
        # data source abort the entire calibration)
        india_vix           = self._safe(india_vix, 16.0, "india_vix")
        nifty_change_pct    = self._safe(nifty_change_pct, 0.0, "nifty_change_pct")
        fii_dii_flow        = self._safe(fii_dii_flow, {}, "fii_dii_flow")
        sector_momentum     = self._safe(sector_momentum, {}, "sector_momentum")
        market_breadth      = self._safe(market_breadth, 1.0, "market_breadth")
        options_iv_percentile = self._safe(options_iv_percentile, 50.0, "options_iv_percentile")

        # Synchronous reads from the portfolio state tracker
        current_drawdown    = self._safe_sync(self._tools.get_current_drawdown,  0.0, "current_drawdown")
        daily_pnl_pct       = self._safe_sync(self._tools.get_daily_pnl_pct,     0.0, "daily_pnl_pct")
        agent_sharpe_30d    = self._safe_sync(self._tools.get_agent_sharpe_scores, {}, "agent_sharpe_30d")

        # Run local regime detection (no LLM, no network)
        regime_result: RegimeResult = self._regime_detector._detect_sync(
            india_vix=india_vix,
            adx=market_breadth if isinstance(market_breadth, float) else 20.0,
            advance_decline_ratio=market_breadth if isinstance(market_breadth, float) else 1.0,
            nifty_change_pct=nifty_change_pct,
            fii_net_flow_cr=(fii_dii_flow or {}).get("fii_net_cr"),
        )

        return {
            "india_vix":              india_vix,
            "nifty_change_pct":       nifty_change_pct,
            "fii_dii_flow":           fii_dii_flow,
            "pre_detected_regime":    regime_result.regime.value,
            "regime_confidence":      regime_result.confidence,
            "current_drawdown":       current_drawdown,
            "daily_pnl_pct":          daily_pnl_pct,
            "agent_sharpe_30d":       agent_sharpe_30d,
            "sector_momentum":        sector_momentum,
            "market_breadth_ad_ratio": market_breadth,
            "options_iv_percentile":  options_iv_percentile,
            "time_of_day":            now_ist.strftime("%H:%M"),
            "day_of_week":            now_ist.strftime("%A"),
            "is_market_hours":        self._is_market_hours(),
        }

    # ── Utilities ─────────────────────────────────────────────────────────

    @staticmethod
    def _safe(value: Any, default: Any, label: str) -> Any:
        if isinstance(value, Exception):
            log.warning("CalibrationAgent: %s fetch failed (%s) — using default", label, value)
            return default
        return value

    @staticmethod
    def _safe_sync(fn, default: Any, label: str) -> Any:
        try:
            return fn()
        except Exception as exc:
            log.warning("CalibrationAgent: %s read failed (%s) — using default", label, exc)
            return default

    @staticmethod
    def _is_market_hours() -> bool:
        now = datetime.now(IST)
        if now.weekday() >= 5:  # Saturday / Sunday
            return False
        open_  = now.replace(hour=_MARKET_OPEN_H,  minute=_MARKET_OPEN_M,  second=0, microsecond=0)
        close_ = now.replace(hour=_MARKET_CLOSE_H, minute=_MARKET_CLOSE_M, second=0, microsecond=0)
        return open_ <= now <= close_

    @staticmethod
    def _build_default_result() -> CalibrationResult:
        """
        Fallback calibration used when both LLM and cache are unavailable.
        Uses conservative static defaults from config.
        """
        from config.strategy_params import (
            DEFAULT_AGENT_WEIGHTS,
            DEFAULT_POSITION_SIZING,
            DEFAULT_RISK_THRESHOLDS,
            DEFAULT_SIGNAL_THRESHOLD,
            DEFAULT_SIGNAL_WEIGHTS,
            DEFAULT_SL_TP,
        )

        raw = {
            "market_regime":   Regime.UNKNOWN.value,
            "signal_weights":  DEFAULT_SIGNAL_WEIGHTS,
            "signal_threshold": DEFAULT_SIGNAL_THRESHOLD,
            "agent_weights":   DEFAULT_AGENT_WEIGHTS,
            "risk_thresholds": DEFAULT_RISK_THRESHOLDS,
            "sl_tp_multipliers": DEFAULT_SL_TP,
            "position_sizing": DEFAULT_POSITION_SIZING,
            "qlib_models": {
                "NSE500":  {"factors": "alpha158_full",      "model": "lightgbm_nse500"},
                "NIFTY50": {"factors": "alpha158_large_cap", "model": "lightgbm_nifty50"},
                "NIFTYIT": {"factors": "alpha158_it_sector", "model": "lightgbm_niftyit"},
                "BSE":     {"factors": "alpha158_bse",       "model": "lightgbm_bse"},
                "BANKING": {"factors": "alpha158_banking",   "model": "lightgbm_banking"},
                "PHARMA":  {"factors": "alpha158_pharma",    "model": "lightgbm_pharma"},
                "default": {"factors": "alpha158_full",      "model": "lightgbm_nse500"},
            },
            "kill_switch": False,
            "reasoning": "Static defaults used (LLM unavailable).",
        }
        return CalibrationResult(
            raw=raw,
            regime=Regime.UNKNOWN,
            cached_at=time.monotonic(),
            reasoning=raw["reasoning"],
        )

    def invalidate_cache(self) -> None:
        """Force the next call to refresh from the LLM. Useful for tests."""
        self._cache = None
        log.debug("CalibrationAgent: cache invalidated")
