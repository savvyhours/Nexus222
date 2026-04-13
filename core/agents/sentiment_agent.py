"""
NEXUS-II — SentimentAgent
News-driven trades using FinBERT scores from the sentiment engine.

Rules:
  - Score > +0.6 sustained 2 hrs → BUY
  - Score < -0.6 sustained 2 hrs → SELL
  - Score swing > 0.4 in 30 min → immediate signal
  - Strength boosted by mention_velocity (mentions/hour)
"""
from __future__ import annotations

import logging

from config.strategy_params import (
    SENTIMENT_BUY_THRESHOLD, SENTIMENT_HOLD_HOURS,
    SENTIMENT_SELL_THRESHOLD, SENTIMENT_SWING_MINUTES,
    SENTIMENT_SWING_THRESHOLD,
)
from core.agents.base_agent import Action, AgentSignal, BaseAgent

log = logging.getLogger(__name__)


class SentimentAgent(BaseAgent):
    """News/social-media sentiment-driven trade signals."""

    AGENT_KEY = "sentiment"

    async def analyze(self, market_data: dict, sentiment_data: dict) -> list[AgentSignal]:
        signals: list[AgentSignal] = []
        symbol = market_data.get("symbol", "UNKNOWN")
        ind    = market_data.get("indicators", {})
        quote  = market_data.get("quote", {})
        ltp    = float(quote.get("ltp", 0.0))
        atr    = float(ind.get("atr", 0.0))

        score           = float(sentiment_data.get("score", 0.0))
        sustained_hours = float(sentiment_data.get("sustained_hours", 0.0))
        swing_30m       = float(sentiment_data.get("swing_30m", 0.0))
        velocity        = float(sentiment_data.get("mention_velocity", 0.0))  # mentions/hour

        mult   = await self._calibration.get_sl_tp_multipliers()
        sizing = await self._calibration.get_position_sizing()
        sl_m   = mult.get("intraday_sl_atr", 2.0)
        rr     = mult.get("target_risk_reward", 2.0)

        # Velocity boosts strength up to +0.20
        vel_boost = min(0.20, velocity / 500 * 0.20) if velocity > 0 else 0.0

        # ── Sustained positive sentiment → BUY ─────────────────────────────
        if score >= SENTIMENT_BUY_THRESHOLD and sustained_hours >= SENTIMENT_HOLD_HOURS:
            strength = min(1.0, 0.55 + (score - SENTIMENT_BUY_THRESHOLD) * 0.5 + vel_boost)
            signals.append(self._make_signal(
                symbol=symbol, action=Action.BUY, strength=round(strength, 3),
                reason=(
                    f"Positive sentiment score={score:.2f} sustained "
                    f"{sustained_hours:.1f}h (need ≥{SENTIMENT_HOLD_HOURS}h). "
                    f"Velocity={velocity:.0f} mentions/hr."
                ),
                strategy="sustained_sentiment", entry=ltp,
                stop_loss=ltp - atr * sl_m, target=ltp + atr * sl_m * rr,
                position_size_pct=sizing.get("default_pct", 0.03),
                metadata={"score": score, "velocity": velocity},
            ))

        # ── Sustained negative sentiment → SELL ────────────────────────────
        elif score <= SENTIMENT_SELL_THRESHOLD and sustained_hours >= SENTIMENT_HOLD_HOURS:
            strength = min(1.0, 0.55 + (abs(score) - abs(SENTIMENT_SELL_THRESHOLD)) * 0.5 + vel_boost)
            signals.append(self._make_signal(
                symbol=symbol, action=Action.SELL, strength=round(strength, 3),
                reason=(
                    f"Negative sentiment score={score:.2f} sustained "
                    f"{sustained_hours:.1f}h (need ≥{SENTIMENT_HOLD_HOURS}h). "
                    f"Velocity={velocity:.0f} mentions/hr."
                ),
                strategy="sustained_sentiment", entry=ltp,
                stop_loss=ltp + atr * sl_m, target=ltp - atr * sl_m * rr,
                position_size_pct=sizing.get("default_pct", 0.03),
                metadata={"score": score, "velocity": velocity},
            ))

        # ── Rapid sentiment swing → immediate signal ────────────────────────
        if abs(swing_30m) >= SENTIMENT_SWING_THRESHOLD:
            action = Action.BUY if swing_30m > 0 else Action.SELL
            strength = min(1.0, abs(swing_30m) / 1.0 * 0.70 + vel_boost)
            sl = ltp - atr * sl_m if action == Action.BUY else ltp + atr * sl_m
            tgt = ltp + atr * sl_m * rr if action == Action.BUY else ltp - atr * sl_m * rr
            signals.append(self._make_signal(
                symbol=symbol, action=action, strength=round(strength, 3),
                reason=(
                    f"Rapid sentiment swing={swing_30m:+.2f} in {SENTIMENT_SWING_MINUTES}min "
                    f"(threshold ≥{SENTIMENT_SWING_THRESHOLD}). Velocity={velocity:.0f}."
                ),
                strategy="sentiment_swing", entry=ltp,
                stop_loss=sl, target=tgt,
                position_size_pct=sizing.get("low_conviction_pct", 0.01),
                metadata={"swing_30m": swing_30m, "velocity": velocity},
            ))

        log.debug("SentimentAgent %s: %d signal(s), score=%.2f", symbol, len(signals), score)
        return signals
