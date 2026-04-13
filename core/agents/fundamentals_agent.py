"""
NEXUS-II — FundamentalsAgent
Value picks driven by Screener.in fundamentals + Qlib factor scores.
Generates medium-term positional signals (days to weeks).
"""
from __future__ import annotations

import logging

from core.agents.base_agent import Action, AgentSignal, BaseAgent

log = logging.getLogger(__name__)

# Fundamental thresholds
_PE_CHEAP  = 15.0    # P/E below this → attractively valued
_PE_DEAR   = 35.0    # P/E above this → expensive
_ROE_MIN   = 15.0    # ROE % minimum for quality
_EPS_GROWTH_MIN = 15.0  # YoY EPS growth % minimum


class FundamentalsAgent(BaseAgent):
    """Value/quality picks using P/E, EPS growth, ROE, Qlib factor scores."""

    AGENT_KEY = "fundamentals"

    async def analyze(self, market_data: dict, sentiment_data: dict) -> list[AgentSignal]:
        signals: list[AgentSignal] = []
        symbol = market_data.get("symbol", "UNKNOWN")
        ind    = market_data.get("indicators", {})
        quote  = market_data.get("quote", {})
        fund   = market_data.get("fundamentals", {})

        ltp        = float(quote.get("ltp", 0.0))
        atr        = float(ind.get("atr", 0.0))
        pe         = float(fund.get("pe", 0.0))
        roe        = float(fund.get("roe", 0.0))
        eps_growth = float(fund.get("eps_growth_yoy", 0.0))
        pb         = float(fund.get("pb", 0.0))
        qlib_score = float(fund.get("qlib_factor_score", 0.0))  # normalised -1..1

        mult   = await self._calibration.get_sl_tp_multipliers()
        sizing = await self._calibration.get_position_sizing()
        sl_m   = mult.get("positional_sl_atr", 3.0)
        rr     = mult.get("target_risk_reward", 2.0)

        # ── Quality + Value BUY ─────────────────────────────────────────────
        if (
            pe > 0 and pe < _PE_CHEAP
            and roe >= _ROE_MIN
            and eps_growth >= _EPS_GROWTH_MIN
        ):
            strength = min(1.0, 0.55
                + ((_PE_CHEAP - pe) / _PE_CHEAP) * 0.20
                + min(0.15, (roe - _ROE_MIN) / 30 * 0.15)
                + min(0.10, qlib_score * 0.10))
            signals.append(self._make_signal(
                symbol=symbol, action=Action.BUY, strength=round(strength, 3),
                reason=(
                    f"Value BUY: P/E={pe:.1f} < {_PE_CHEAP}, ROE={roe:.1f}%, "
                    f"EPS growth={eps_growth:.1f}%, Qlib score={qlib_score:.2f}"
                ),
                strategy="fundamental_value", entry=ltp,
                stop_loss=ltp - atr * sl_m, target=ltp + atr * sl_m * rr,
                position_size_pct=sizing.get("default_pct", 0.03),
                metadata={"pe": pe, "roe": roe, "eps_growth": eps_growth},
            ))

        # ── Expensive / deteriorating fundamentals → SELL ──────────────────
        elif pe > _PE_DEAR and roe < _ROE_MIN * 0.5 and eps_growth < 0:
            signals.append(self._make_signal(
                symbol=symbol, action=Action.SELL, strength=0.55,
                reason=(
                    f"Overvalued: P/E={pe:.1f} > {_PE_DEAR}, ROE={roe:.1f}%, "
                    f"EPS growth={eps_growth:.1f}% (negative)"
                ),
                strategy="fundamental_overvalued", entry=ltp,
                stop_loss=ltp + atr * sl_m, target=ltp - atr * sl_m * rr,
                position_size_pct=sizing.get("low_conviction_pct", 0.01),
                metadata={"pe": pe, "roe": roe, "eps_growth": eps_growth},
            ))

        # ── High Qlib factor score → directional signal ─────────────────────
        if abs(qlib_score) >= 0.65:
            action = Action.BUY if qlib_score > 0 else Action.SELL
            sl  = ltp - atr * sl_m if action == Action.BUY else ltp + atr * sl_m
            tgt = ltp + atr * sl_m * rr if action == Action.BUY else ltp - atr * sl_m * rr
            signals.append(self._make_signal(
                symbol=symbol, action=action,
                strength=min(1.0, abs(qlib_score)),
                reason=f"Qlib Alpha158 factor score={qlib_score:.2f} (|score|≥0.65)",
                strategy="qlib_factor", entry=ltp, stop_loss=sl, target=tgt,
                position_size_pct=sizing.get("default_pct", 0.03),
                metadata={"qlib_score": qlib_score},
            ))

        log.debug("FundamentalsAgent %s: %d signal(s)", symbol, len(signals))
        return signals
