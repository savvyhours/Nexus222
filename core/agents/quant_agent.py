"""
NEXUS-II — QuantAgent
Quantitative / ML-driven signals using Qlib Alpha158 factor rankings.

Logic:
  - Top-decile Qlib rank (buy candidates) → BUY
  - Bottom-decile rank (short candidates) → SELL
  - Also combines: Momentum factor (20-day return rank), Quality, Value, Low Vol
"""
from __future__ import annotations

import logging

from core.agents.base_agent import Action, AgentSignal, BaseAgent

log = logging.getLogger(__name__)

_TOP_DECILE_PERCENTILE    = 90  # rank ≥ 90th → BUY
_BOTTOM_DECILE_PERCENTILE = 10  # rank ≤ 10th → SELL


class QuantAgent(BaseAgent):
    """Qlib Alpha158 + LightGBM factor model signal generator."""

    AGENT_KEY = "quant"

    async def analyze(self, market_data: dict, sentiment_data: dict) -> list[AgentSignal]:
        signals: list[AgentSignal] = []
        symbol = market_data.get("symbol", "UNKNOWN")
        ind    = market_data.get("indicators", {})
        quote  = market_data.get("quote", {})
        fund   = market_data.get("fundamentals", {})

        ltp = float(quote.get("ltp", 0.0))
        atr = float(ind.get("atr", 0.0))

        # Qlib outputs
        qlib_rank_pct  = float(fund.get("qlib_rank_percentile", 50.0))   # 0-100
        pred_return_5d = float(fund.get("qlib_predicted_return_5d", 0.0)) # predicted 5-day return
        momentum_rank  = float(fund.get("momentum_20d_rank", 50.0))       # 0-100 percentile
        quality_score  = float(fund.get("quality_score", 0.5))            # 0-1
        low_vol_rank   = float(fund.get("low_vol_rank", 50.0))            # higher = lower vol

        mult   = await self._calibration.get_sl_tp_multipliers()
        sizing = await self._calibration.get_position_sizing()
        sl_m   = mult.get("positional_sl_atr", 3.0)
        rr     = mult.get("target_risk_reward", 2.0)

        # Composite score: weighted average of all factor ranks
        composite = (
            qlib_rank_pct * 0.40
            + momentum_rank * 0.30
            + quality_score * 100 * 0.20
            + low_vol_rank * 0.10
        )

        # ── Top decile → BUY ─────────────────────────────────────────────────
        if qlib_rank_pct >= _TOP_DECILE_PERCENTILE and pred_return_5d > 0:
            strength = min(1.0, 0.55 + (qlib_rank_pct - _TOP_DECILE_PERCENTILE) / 10 * 0.25
                        + min(0.15, pred_return_5d * 3))
            signals.append(self._make_signal(
                symbol=symbol, action=Action.BUY, strength=round(strength, 3),
                reason=(
                    f"Qlib rank={qlib_rank_pct:.0f}th pct (top decile), "
                    f"pred 5d return={pred_return_5d:.2%}, "
                    f"momentum rank={momentum_rank:.0f}th"
                ),
                strategy="qlib_top_decile", entry=ltp,
                stop_loss=ltp - atr * sl_m, target=ltp + atr * sl_m * rr,
                position_size_pct=sizing.get("default_pct", 0.03),
                metadata={
                    "qlib_rank": qlib_rank_pct,
                    "pred_return_5d": pred_return_5d,
                    "composite": composite,
                },
            ))

        # ── Bottom decile → SELL ──────────────────────────────────────────────
        elif qlib_rank_pct <= _BOTTOM_DECILE_PERCENTILE and pred_return_5d < 0:
            strength = min(1.0, 0.55 + (_BOTTOM_DECILE_PERCENTILE - qlib_rank_pct) / 10 * 0.25
                        + min(0.15, abs(pred_return_5d) * 3))
            signals.append(self._make_signal(
                symbol=symbol, action=Action.SELL, strength=round(strength, 3),
                reason=(
                    f"Qlib rank={qlib_rank_pct:.0f}th pct (bottom decile), "
                    f"pred 5d return={pred_return_5d:.2%}, "
                    f"momentum rank={momentum_rank:.0f}th"
                ),
                strategy="qlib_bottom_decile", entry=ltp,
                stop_loss=ltp + atr * sl_m, target=ltp - atr * sl_m * rr,
                position_size_pct=sizing.get("default_pct", 0.03),
                metadata={
                    "qlib_rank": qlib_rank_pct,
                    "pred_return_5d": pred_return_5d,
                    "composite": composite,
                },
            ))

        # ── Strong momentum + quality BUY ─────────────────────────────────────
        if momentum_rank >= 85 and quality_score >= 0.7 and qlib_rank_pct >= 70:
            signals.append(self._make_signal(
                symbol=symbol, action=Action.BUY, strength=0.72,
                reason=(
                    f"High momentum ({momentum_rank:.0f}th pct) + quality ({quality_score:.2f}) + "
                    f"Qlib rank ({qlib_rank_pct:.0f}th pct)"
                ),
                strategy="momentum_quality", entry=ltp,
                stop_loss=ltp - atr * sl_m, target=ltp + atr * sl_m * rr,
                position_size_pct=sizing.get("default_pct", 0.03),
            ))

        log.debug("QuantAgent %s: %d signal(s), Qlib_rank=%.0f", symbol, len(signals), qlib_rank_pct)
        return signals
