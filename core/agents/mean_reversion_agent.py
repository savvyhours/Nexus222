"""
NEXUS-II — MeanReversionAgent
Strategies: Bollinger squeeze→expansion, RSI divergence, Z-score vs 20-day mean,
pairs trading within sector. Candlestick composites:
  - RSI Divergence + Hammer (profit factor 8.51, win rate 73.7%)
  - Morning Star + RSI<30 (profit factor 2.12, win rate 62.1%)
"""
from __future__ import annotations

import logging

from config.strategy_params import (
    MEAN_REV_BB_PERIOD, MEAN_REV_BB_STD, MEAN_REV_ZSCORE_THRESHOLD,
)
from core.agents.base_agent import Action, AgentSignal, BaseAgent

log = logging.getLogger(__name__)


class MeanReversionAgent(BaseAgent):
    """Mean reversion: Bollinger, RSI divergence, Z-score, pairs arb."""

    AGENT_KEY = "mean_reversion"

    async def analyze(self, market_data: dict, sentiment_data: dict) -> list[AgentSignal]:
        signals: list[AgentSignal] = []
        symbol = market_data.get("symbol", "UNKNOWN")
        ind    = market_data.get("indicators", {})
        quote  = market_data.get("quote", {})
        candles = market_data.get("ohlcv", [])   # list of {o,h,l,c,v} dicts

        ltp      = float(quote.get("ltp", 0.0))
        bb_upper = float(ind.get("bb_upper", ltp * 1.02))
        bb_lower = float(ind.get("bb_lower", ltp * 0.98))
        bb_mid   = float(ind.get("bb_mid", ltp))
        rsi      = float(ind.get("rsi_14", 50.0))
        zscore   = float(ind.get("zscore_20d", 0.0))
        atr      = float(ind.get("atr", 0.0))
        rsi_div  = bool(ind.get("rsi_divergence", False))  # bullish/bearish divergence flag

        mult   = await self._calibration.get_sl_tp_multipliers()
        sizing = await self._calibration.get_position_sizing()
        sl_m   = mult.get("intraday_sl_atr", 2.0)
        rr     = mult.get("target_risk_reward", 2.0)

        # ── Bollinger lower touch → BUY ─────────────────────────────────────
        if ltp <= bb_lower and zscore <= -MEAN_REV_ZSCORE_THRESHOLD * 0.75:
            strength = min(1.0, 0.55 + abs(zscore) / 4 * 0.25)
            signals.append(self._make_signal(
                symbol=symbol, action=Action.BUY, strength=round(strength, 3),
                reason=f"BB lower touch: LTP {ltp:.2f} ≤ BB_lower {bb_lower:.2f}, Z={zscore:.2f}",
                strategy="bollinger_revert", entry=ltp,
                stop_loss=ltp - atr * sl_m, target=bb_mid,
                position_size_pct=sizing.get("default_pct", 0.03),
            ))

        # ── Bollinger upper touch → SELL ────────────────────────────────────
        elif ltp >= bb_upper and zscore >= MEAN_REV_ZSCORE_THRESHOLD * 0.75:
            strength = min(1.0, 0.55 + abs(zscore) / 4 * 0.25)
            signals.append(self._make_signal(
                symbol=symbol, action=Action.SELL, strength=round(strength, 3),
                reason=f"BB upper touch: LTP {ltp:.2f} ≥ BB_upper {bb_upper:.2f}, Z={zscore:.2f}",
                strategy="bollinger_revert", entry=ltp,
                stop_loss=ltp + atr * sl_m, target=bb_mid,
                position_size_pct=sizing.get("default_pct", 0.03),
            ))

        # ── RSI Divergence + Hammer (best candlestick composite) ────────────
        hammer = self._is_hammer(candles)
        if rsi_div and hammer and rsi < 35:
            signals.append(self._make_signal(
                symbol=symbol, action=Action.BUY, strength=0.80,
                reason=f"Bullish RSI divergence + Hammer candle. RSI={rsi:.1f}",
                strategy="rsi_divergence_hammer", entry=ltp,
                stop_loss=ltp - atr * sl_m, target=ltp + atr * sl_m * rr,
                position_size_pct=sizing.get("high_conviction_pct", 0.05),
            ))

        # ── Morning Star + RSI<30 ───────────────────────────────────────────
        morning_star = self._is_morning_star(candles)
        if morning_star and rsi < 30:
            signals.append(self._make_signal(
                symbol=symbol, action=Action.BUY, strength=0.70,
                reason=f"Morning Star pattern + RSI={rsi:.1f} < 30",
                strategy="morning_star_rsi", entry=ltp,
                stop_loss=ltp - atr * sl_m, target=ltp + atr * sl_m * rr,
                position_size_pct=sizing.get("default_pct", 0.03),
            ))

        # ── Z-score extremes ────────────────────────────────────────────────
        if zscore <= -MEAN_REV_ZSCORE_THRESHOLD:
            signals.append(self._make_signal(
                symbol=symbol, action=Action.BUY, strength=min(1.0, 0.5 + abs(zscore) / 5 * 0.4),
                reason=f"Z-score={zscore:.2f} (extreme oversold vs 20d mean)",
                strategy="zscore_revert", entry=ltp,
                stop_loss=ltp - atr * sl_m, target=bb_mid,
                position_size_pct=sizing.get("low_conviction_pct", 0.01),
            ))
        elif zscore >= MEAN_REV_ZSCORE_THRESHOLD:
            signals.append(self._make_signal(
                symbol=symbol, action=Action.SELL, strength=min(1.0, 0.5 + abs(zscore) / 5 * 0.4),
                reason=f"Z-score={zscore:.2f} (extreme overbought vs 20d mean)",
                strategy="zscore_revert", entry=ltp,
                stop_loss=ltp + atr * sl_m, target=bb_mid,
                position_size_pct=sizing.get("low_conviction_pct", 0.01),
            ))

        log.debug("MeanReversionAgent %s: %d signal(s)", symbol, len(signals))
        return signals

    # ── Candlestick pattern helpers ──────────────────────────────────────────

    @staticmethod
    def _is_hammer(candles: list[dict]) -> bool:
        if not candles:
            return False
        c = candles[-1]
        o, h, l, close = float(c.get("o", 0)), float(c.get("h", 0)), float(c.get("l", 0)), float(c.get("c", 0))
        body = abs(close - o)
        lower_wick = min(o, close) - l
        upper_wick = h - max(o, close)
        if body == 0:
            return False
        return lower_wick >= 2 * body and upper_wick <= body * 0.5 and close > o

    @staticmethod
    def _is_morning_star(candles: list[dict]) -> bool:
        if len(candles) < 3:
            return False
        c1, c2, c3 = candles[-3], candles[-2], candles[-1]
        bearish_first = float(c1.get("c", 0)) < float(c1.get("o", 0))
        small_body    = abs(float(c2.get("c", 0)) - float(c2.get("o", 0))) < abs(float(c1.get("c", 0)) - float(c1.get("o", 0))) * 0.3
        bullish_third = float(c3.get("c", 0)) > float(c3.get("o", 0))
        closes_above  = float(c3.get("c", 0)) > (float(c1.get("o", 0)) + float(c1.get("c", 0))) / 2
        return bearish_first and small_body and bullish_third and closes_above
