"""
NEXUS-II — MacroAgent
Index/ETF positioning driven by macro flows: FII/DII, India VIX, USD/INR, crude.

Rules:
  - FII buying 3 consecutive days → NIFTY50/NIFTYBEES BUY
  - VIX > 22 → reduce equity, buy GOLDBEES hedge
  - VIX < 14 → add equity exposure
  - USD/INR > 85 → sell IT sector (NIFTYIT ETF)
"""
from __future__ import annotations

import logging

from config.strategy_params import OPTIONS_VIX_HEDGE_TRIGGER
from core.agents.base_agent import Action, AgentSignal, BaseAgent

log = logging.getLogger(__name__)

_VIX_ADD_EQUITY = 14.0
_VIX_DEFENSIVE  = 22.0
_USDINR_IT_SELL = 85.0
_FII_BUY_STREAK = 3       # consecutive FII net-buy days to trigger
_FII_BUY_MIN_CR = 500.0   # ₹ crore per day net buy threshold


class MacroAgent(BaseAgent):
    """Macro flow-driven index/ETF positioning."""

    AGENT_KEY = "macro"

    async def analyze(self, market_data: dict, sentiment_data: dict) -> list[AgentSignal]:
        signals: list[AgentSignal] = []
        symbol = market_data.get("symbol", "UNKNOWN")
        macro  = market_data.get("macro", {})
        quote  = market_data.get("quote", {})
        ind    = market_data.get("indicators", {})

        ltp         = float(quote.get("ltp", 0.0))
        atr         = float(ind.get("atr", 0.0))
        vix         = float(macro.get("india_vix", 16.0))
        usdinr      = float(macro.get("usdinr", 83.0))
        fii_streak  = int(macro.get("fii_buy_streak_days", 0))    # +n = n consec buy days
        fii_net_cr  = float(macro.get("fii_net_cr_today", 0.0))   # ₹ crore today
        crude_usd   = float(macro.get("crude_usd", 75.0))

        mult   = await self._calibration.get_sl_tp_multipliers()
        sizing = await self._calibration.get_position_sizing()
        rt     = await self._calibration.get_risk_thresholds()
        sl_m   = mult.get("positional_sl_atr", 3.0)
        rr     = mult.get("target_risk_reward", 2.0)
        vix_def = float(rt.get("vix_defensive_threshold", _VIX_DEFENSIVE))

        # ── FII buying streak → NIFTY BUY ───────────────────────────────────
        if fii_streak >= _FII_BUY_STREAK and fii_net_cr >= _FII_BUY_MIN_CR:
            strength = min(1.0, 0.60 + min(0.20, (fii_streak - _FII_BUY_STREAK) * 0.05))
            signals.append(self._make_signal(
                symbol=symbol, action=Action.BUY, strength=round(strength, 3),
                reason=(
                    f"FII buying {fii_streak} consecutive days, "
                    f"today net +₹{fii_net_cr:.0f}cr → broad market bullish"
                ),
                strategy="fii_momentum", entry=ltp,
                stop_loss=ltp - atr * sl_m, target=ltp + atr * sl_m * rr,
                position_size_pct=sizing.get("default_pct", 0.03),
                metadata={"fii_streak": fii_streak, "fii_net_cr": fii_net_cr},
            ))

        # ── VIX > defensive threshold → GOLDBEES hedge ──────────────────────
        if vix >= vix_def:
            signals.append(self._make_signal(
                symbol="GOLDBEES", action=Action.BUY, strength=min(1.0, 0.50 + (vix - vix_def) / 15 * 0.35),
                reason=f"VIX={vix:.1f} ≥ defensive threshold {vix_def:.0f} → GOLDBEES hedge",
                strategy="vix_hedge", entry=0.0, stop_loss=0.0, target=0.0,
                position_size_pct=sizing.get("low_conviction_pct", 0.01),
                metadata={"vix": vix},
            ))
            # Also signal to reduce equity position on current symbol
            if symbol not in ("GOLDBEES", "LIQUIDBEES"):
                signals.append(self._make_signal(
                    symbol=symbol, action=Action.SELL, strength=0.50,
                    reason=f"VIX={vix:.1f} ≥ {vix_def:.0f}: defensive — reduce equity",
                    strategy="vix_defensive", entry=ltp,
                    stop_loss=ltp + atr * sl_m, target=ltp - atr * sl_m,
                    position_size_pct=sizing.get("low_conviction_pct", 0.01),
                ))

        # ── VIX < 14 (complacency) → add equity ─────────────────────────────
        elif vix < _VIX_ADD_EQUITY:
            signals.append(self._make_signal(
                symbol=symbol, action=Action.BUY, strength=0.55,
                reason=f"Low volatility: VIX={vix:.1f} < {_VIX_ADD_EQUITY} → add equity exposure",
                strategy="low_vol_equity", entry=ltp,
                stop_loss=ltp - atr * sl_m, target=ltp + atr * sl_m * rr,
                position_size_pct=sizing.get("default_pct", 0.03),
            ))

        # ── USD/INR > 85 → sell IT sector ────────────────────────────────────
        if usdinr > _USDINR_IT_SELL and "IT" in symbol.upper():
            signals.append(self._make_signal(
                symbol=symbol, action=Action.SELL, strength=0.60,
                reason=f"USD/INR={usdinr:.2f} > {_USDINR_IT_SELL} → IT margin pressure, sell IT",
                strategy="usdinr_it_hedge", entry=ltp,
                stop_loss=ltp + atr * sl_m, target=ltp - atr * sl_m * rr,
                position_size_pct=sizing.get("default_pct", 0.02),
                metadata={"usdinr": usdinr},
            ))

        log.debug("MacroAgent %s: %d signal(s), VIX=%.1f", symbol, len(signals), vix)
        return signals
