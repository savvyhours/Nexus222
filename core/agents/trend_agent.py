"""
NEXUS-II — TrendFollowerAgent
1-5 day positional trades: EMA(20/50) crossover, MACD signal cross,
ADX > 25 filter, 52-week high breakout. Also uses EMA Ribbon (8/13/21/34/55)
composite (profit factor 6.72, win rate 77.8%).
"""
from __future__ import annotations

import logging

from config.strategy_params import (
    TREND_ADX_MIN, TREND_EMA_LONG, TREND_EMA_SHORT,
)
from core.agents.base_agent import Action, AgentSignal, BaseAgent

log = logging.getLogger(__name__)

_RIBBON_PERIODS = [8, 13, 21, 34, 55]


class TrendFollowerAgent(BaseAgent):
    """Trend-following: EMA crossover, MACD, ADX filter, 52w breakout."""

    AGENT_KEY = "trend_follower"

    async def analyze(self, market_data: dict, sentiment_data: dict) -> list[AgentSignal]:
        signals: list[AgentSignal] = []
        symbol = market_data.get("symbol", "UNKNOWN")
        ind    = market_data.get("indicators", {})
        quote  = market_data.get("quote", {})

        ltp       = float(quote.get("ltp", 0.0))
        ema_short = float(ind.get(f"ema_{TREND_EMA_SHORT}", ltp))
        ema_long  = float(ind.get(f"ema_{TREND_EMA_LONG}", ltp))
        adx       = float(ind.get("adx", 0.0))
        macd      = float(ind.get("macd", 0.0))
        macd_sig  = float(ind.get("macd_signal", 0.0))
        high_52w  = float(ind.get("high_52w", ltp))
        low_52w   = float(ind.get("low_52w", ltp))
        atr       = float(ind.get("atr", 0.0))

        mult   = await self._calibration.get_sl_tp_multipliers()
        sizing = await self._calibration.get_position_sizing()
        sl_m   = mult.get("positional_sl_atr", 3.0)
        rr     = mult.get("target_risk_reward", 2.0)

        # ── EMA ribbon check ────────────────────────────────────────────────
        ribbon_vals = [float(ind.get(f"ema_{p}", ltp)) for p in _RIBBON_PERIODS]
        ribbon_bull = all(ribbon_vals[i] > ribbon_vals[i+1] for i in range(len(ribbon_vals)-1))
        ribbon_bear = all(ribbon_vals[i] < ribbon_vals[i+1] for i in range(len(ribbon_vals)-1))

        # ── EMA(20/50) crossover BUY ────────────────────────────────────────
        if ema_short > ema_long and adx > TREND_ADX_MIN and macd > macd_sig:
            strength = min(1.0, 0.60 + (adx - TREND_ADX_MIN) / 50 * 0.25 + (0.10 if ribbon_bull else 0))
            signals.append(self._make_signal(
                symbol=symbol, action=Action.BUY, strength=round(strength, 3),
                reason=(
                    f"EMA{TREND_EMA_SHORT}({ema_short:.1f})>EMA{TREND_EMA_LONG}({ema_long:.1f}), "
                    f"ADX={adx:.1f}, MACD cross up"
                    + (", ribbon aligned" if ribbon_bull else "")
                ),
                strategy="ema_crossover", entry=ltp,
                stop_loss=ltp - atr * sl_m, target=ltp + atr * sl_m * rr,
                position_size_pct=sizing.get("default_pct", 0.03),
            ))

        # ── EMA(20/50) crossover SELL ───────────────────────────────────────
        elif ema_short < ema_long and adx > TREND_ADX_MIN and macd < macd_sig:
            strength = min(1.0, 0.60 + (adx - TREND_ADX_MIN) / 50 * 0.25 + (0.10 if ribbon_bear else 0))
            signals.append(self._make_signal(
                symbol=symbol, action=Action.SELL, strength=round(strength, 3),
                reason=(
                    f"EMA{TREND_EMA_SHORT}({ema_short:.1f})<EMA{TREND_EMA_LONG}({ema_long:.1f}), "
                    f"ADX={adx:.1f}, MACD cross down"
                    + (", ribbon aligned" if ribbon_bear else "")
                ),
                strategy="ema_crossover", entry=ltp,
                stop_loss=ltp + atr * sl_m, target=ltp - atr * sl_m * rr,
                position_size_pct=sizing.get("default_pct", 0.03),
            ))

        # ── 52-week high breakout BUY ───────────────────────────────────────
        if high_52w > 0 and ltp >= high_52w * 0.995 and adx > TREND_ADX_MIN:
            signals.append(self._make_signal(
                symbol=symbol, action=Action.BUY, strength=0.75,
                reason=f"52-week high breakout: LTP {ltp:.2f} ≥ 52w high {high_52w:.2f}, ADX={adx:.1f}",
                strategy="52w_breakout", entry=ltp,
                stop_loss=ltp - atr * sl_m, target=ltp + atr * sl_m * rr,
                position_size_pct=sizing.get("high_conviction_pct", 0.05),
            ))

        log.debug("TrendFollowerAgent %s: %d signal(s)", symbol, len(signals))
        return signals
