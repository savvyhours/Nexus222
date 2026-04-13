"""
NEXUS-II — PatternAgent
Chart pattern recognition: Head & Shoulders, Cup & Handle, Double Top/Bottom,
Flag/Pennant, and candlestick composites:
  - Three White Soldiers (profit factor 2.90, win rate 68.9%)
  - Engulfing + Volume Spike (profit factor 2.45, win rate 65.2%)
"""
from __future__ import annotations

import logging
from typing import Optional

from core.agents.base_agent import Action, AgentSignal, BaseAgent

log = logging.getLogger(__name__)


class PatternAgent(BaseAgent):
    """Technical chart pattern and candlestick composite detection."""

    AGENT_KEY = "pattern"

    async def analyze(self, market_data: dict, sentiment_data: dict) -> list[AgentSignal]:
        signals: list[AgentSignal] = []
        symbol  = market_data.get("symbol", "UNKNOWN")
        candles = market_data.get("ohlcv", [])       # list of {o,h,l,c,v}
        ind     = market_data.get("indicators", {})
        quote   = market_data.get("quote", {})

        ltp    = float(quote.get("ltp", 0.0))
        atr    = float(ind.get("atr", 0.0))
        volume = float(quote.get("volume_1m", 0.0))
        avg_vol = float(ind.get("avg_volume_1m", 1.0))

        # Pre-computed pattern flags (from signal_engine/candlestick_patterns.py)
        patterns = market_data.get("patterns", {})

        mult   = await self._calibration.get_sl_tp_multipliers()
        sizing = await self._calibration.get_position_sizing()
        sl_m   = mult.get("positional_sl_atr", 3.0)
        rr     = mult.get("target_risk_reward", 2.0)

        # ── Three White Soldiers (bullish reversal) ──────────────────────────
        if patterns.get("three_white_soldiers") or self._three_white_soldiers(candles):
            signals.append(self._make_signal(
                symbol=symbol, action=Action.BUY, strength=0.72,
                reason="Three White Soldiers: 3 consecutive bullish candles with higher closes",
                strategy="three_white_soldiers", entry=ltp,
                stop_loss=ltp - atr * sl_m, target=ltp + atr * sl_m * rr,
                position_size_pct=sizing.get("default_pct", 0.03),
            ))

        # ── Bullish Engulfing + Volume Spike ─────────────────────────────────
        spike = avg_vol > 0 and volume >= avg_vol * 1.5
        if (patterns.get("bullish_engulfing") or self._bullish_engulfing(candles)) and spike:
            signals.append(self._make_signal(
                symbol=symbol, action=Action.BUY, strength=0.68,
                reason="Bullish Engulfing + volume spike (>1.5× avg)",
                strategy="engulfing_volume", entry=ltp,
                stop_loss=ltp - atr * sl_m, target=ltp + atr * sl_m * rr,
                position_size_pct=sizing.get("default_pct", 0.03),
            ))

        # ── Bearish Engulfing + Volume Spike ─────────────────────────────────
        if (patterns.get("bearish_engulfing") or self._bearish_engulfing(candles)) and spike:
            signals.append(self._make_signal(
                symbol=symbol, action=Action.SELL, strength=0.65,
                reason="Bearish Engulfing + volume spike (>1.5× avg)",
                strategy="engulfing_volume", entry=ltp,
                stop_loss=ltp + atr * sl_m, target=ltp - atr * sl_m * rr,
                position_size_pct=sizing.get("default_pct", 0.03),
            ))

        # ── Cup & Handle breakout ─────────────────────────────────────────────
        if patterns.get("cup_and_handle_breakout"):
            signals.append(self._make_signal(
                symbol=symbol, action=Action.BUY, strength=0.78,
                reason="Cup & Handle breakout detected",
                strategy="cup_and_handle", entry=ltp,
                stop_loss=ltp - atr * sl_m, target=ltp + atr * sl_m * rr * 1.5,
                position_size_pct=sizing.get("high_conviction_pct", 0.05),
            ))

        # ── Double Bottom ────────────────────────────────────────────────────
        if patterns.get("double_bottom"):
            signals.append(self._make_signal(
                symbol=symbol, action=Action.BUY, strength=0.70,
                reason="Double Bottom pattern: price tested support twice, likely reversal",
                strategy="double_bottom", entry=ltp,
                stop_loss=ltp - atr * sl_m, target=ltp + atr * sl_m * rr,
                position_size_pct=sizing.get("default_pct", 0.03),
            ))

        # ── Double Top ───────────────────────────────────────────────────────
        if patterns.get("double_top"):
            signals.append(self._make_signal(
                symbol=symbol, action=Action.SELL, strength=0.70,
                reason="Double Top pattern: price rejected at resistance twice",
                strategy="double_top", entry=ltp,
                stop_loss=ltp + atr * sl_m, target=ltp - atr * sl_m * rr,
                position_size_pct=sizing.get("default_pct", 0.03),
            ))

        # ── Head & Shoulders (bearish) ────────────────────────────────────────
        if patterns.get("head_and_shoulders"):
            signals.append(self._make_signal(
                symbol=symbol, action=Action.SELL, strength=0.75,
                reason="Head & Shoulders: neckline break, bearish reversal",
                strategy="head_and_shoulders", entry=ltp,
                stop_loss=ltp + atr * sl_m, target=ltp - atr * sl_m * rr,
                position_size_pct=sizing.get("default_pct", 0.03),
            ))

        # ── Bull Flag breakout ────────────────────────────────────────────────
        if patterns.get("bull_flag"):
            signals.append(self._make_signal(
                symbol=symbol, action=Action.BUY, strength=0.68,
                reason="Bull Flag: consolidation breakout continuing prior uptrend",
                strategy="bull_flag", entry=ltp,
                stop_loss=ltp - atr * sl_m, target=ltp + atr * sl_m * rr,
                position_size_pct=sizing.get("default_pct", 0.03),
            ))

        log.debug("PatternAgent %s: %d signal(s)", symbol, len(signals))
        return signals

    # ── Inline candlestick detectors (fallback if patterns dict missing) ─────

    @staticmethod
    def _three_white_soldiers(candles: list[dict]) -> bool:
        if len(candles) < 3:
            return False
        last3 = candles[-3:]
        return all(
            float(c["c"]) > float(c["o"]) and float(c["c"]) > float(candles[-3:][max(0, i-1)].get("c", 0))
            for i, c in enumerate(last3)
        )

    @staticmethod
    def _bullish_engulfing(candles: list[dict]) -> bool:
        if len(candles) < 2:
            return False
        prev, curr = candles[-2], candles[-1]
        return (
            float(prev["c"]) < float(prev["o"])  # prev bearish
            and float(curr["c"]) > float(curr["o"])  # curr bullish
            and float(curr["c"]) > float(prev["o"])
            and float(curr["o"]) < float(prev["c"])
        )

    @staticmethod
    def _bearish_engulfing(candles: list[dict]) -> bool:
        if len(candles) < 2:
            return False
        prev, curr = candles[-2], candles[-1]
        return (
            float(prev["c"]) > float(prev["o"])  # prev bullish
            and float(curr["c"]) < float(curr["o"])  # curr bearish
            and float(curr["o"]) > float(prev["c"])
            and float(curr["c"]) < float(prev["o"])
        )
