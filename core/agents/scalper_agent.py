"""
NEXUS-II — ScalperAgent
1-5 min intraday scalping: VWAP crossover, volume spike, RSI extremes, EMA crossover.
"""
from __future__ import annotations

import logging

from config.strategy_params import (
    SCALPER_EMA_FAST, SCALPER_EMA_SLOW, SCALPER_RSI_PERIOD,
    SCALPER_VOLUME_SPIKE_MULTIPLIER,
)
from core.agents.base_agent import Action, AgentSignal, BaseAgent

log = logging.getLogger(__name__)


class ScalperAgent(BaseAgent):
    """High-frequency intraday scalping on top-20 NSE liquid stocks."""

    AGENT_KEY = "scalper"

    async def analyze(self, market_data: dict, sentiment_data: dict) -> list[AgentSignal]:
        signals: list[AgentSignal] = []
        symbol = market_data.get("symbol", "UNKNOWN")
        ind    = market_data.get("indicators", {})
        quote  = market_data.get("quote", {})

        ltp        = float(quote.get("ltp", 0.0))
        volume_1m  = float(quote.get("volume_1m", 0.0))
        avg_volume = float(ind.get("avg_volume_1m", 1.0))
        rsi        = float(ind.get(f"rsi_{SCALPER_RSI_PERIOD}", 50.0))
        ema_fast   = float(ind.get(f"ema_{SCALPER_EMA_FAST}", ltp))
        ema_slow   = float(ind.get(f"ema_{SCALPER_EMA_SLOW}", ltp))
        vwap       = float(ind.get("vwap", ltp))
        atr        = float(ind.get("atr", 0.0))

        thresholds = await self._calibration.get_risk_thresholds()
        if volume_1m < thresholds.get("min_liquidity_volume", 50_000):
            return []

        mult   = await self._calibration.get_sl_tp_multipliers()
        sizing = await self._calibration.get_position_sizing()
        sl_m   = mult.get("intraday_sl_atr", 2.0)
        rr     = mult.get("target_risk_reward", 2.0)
        spike  = avg_volume > 0 and volume_1m >= avg_volume * SCALPER_VOLUME_SPIKE_MULTIPLIER

        # VWAP + EMA BUY
        if ltp > vwap and ema_fast > ema_slow and spike and rsi < 65:
            signals.append(self._make_signal(
                symbol=symbol, action=Action.BUY,
                strength=min(1.0, 0.55 + (0.15 if spike else 0) + max(0, (65 - rsi) / 65 * 0.15)),
                reason=f"VWAP crossover up, EMA{SCALPER_EMA_FAST}>{SCALPER_EMA_SLOW}, vol spike, RSI={rsi:.1f}",
                strategy="vwap_crossover", entry=ltp,
                stop_loss=ltp - atr * sl_m, target=ltp + atr * sl_m * rr,
                position_size_pct=sizing.get("default_pct", 0.02),
            ))

        # VWAP + EMA SELL
        elif ltp < vwap and ema_fast < ema_slow and spike and rsi > 35:
            signals.append(self._make_signal(
                symbol=symbol, action=Action.SELL,
                strength=min(1.0, 0.55 + (0.15 if spike else 0) + max(0, (rsi - 35) / 65 * 0.15)),
                reason=f"VWAP crossover down, EMA{SCALPER_EMA_FAST}<{SCALPER_EMA_SLOW}, vol spike, RSI={rsi:.1f}",
                strategy="vwap_crossover", entry=ltp,
                stop_loss=ltp + atr * sl_m, target=ltp - atr * sl_m * rr,
                position_size_pct=sizing.get("default_pct", 0.02),
            ))

        # RSI oversold BUY
        if rsi < 30 and spike:
            signals.append(self._make_signal(
                symbol=symbol, action=Action.BUY,
                strength=min(1.0, (30 - rsi) / 30 * 0.8 + 0.4),
                reason=f"RSI({SCALPER_RSI_PERIOD}) oversold={rsi:.1f}, volume spike",
                strategy="rsi_extreme", entry=ltp,
                stop_loss=ltp - atr * sl_m, target=ltp + atr * sl_m * 1.5,
                position_size_pct=sizing.get("low_conviction_pct", 0.01),
            ))

        # RSI overbought SELL
        elif rsi > 70 and spike:
            signals.append(self._make_signal(
                symbol=symbol, action=Action.SELL,
                strength=min(1.0, (rsi - 70) / 30 * 0.8 + 0.4),
                reason=f"RSI({SCALPER_RSI_PERIOD}) overbought={rsi:.1f}, volume spike",
                strategy="rsi_extreme", entry=ltp,
                stop_loss=ltp + atr * sl_m, target=ltp - atr * sl_m * 1.5,
                position_size_pct=sizing.get("low_conviction_pct", 0.01),
            ))

        log.debug("ScalperAgent %s: %d signal(s)", symbol, len(signals))
        return signals
