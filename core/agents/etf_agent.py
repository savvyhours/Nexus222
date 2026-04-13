"""
NEXUS-II — ETFAgent
ETF/Index strategies: NAV premium/discount arb, sector rotation (macro-based),
monthly momentum rebalance, safe haven rotation (equity ↔ gold ↔ liquid by VIX).
"""
from __future__ import annotations

import logging

from core.agents.base_agent import Action, AgentSignal, BaseAgent

log = logging.getLogger(__name__)

# ETF universe
_EQUITY_ETF  = "NIFTYBEES"
_GOLD_ETF    = "GOLDBEES"
_LIQUID_ETF  = "LIQUIDBEES"
_BANK_ETF    = "BANKBEES"
_IT_ETF      = "ICICINIFTY"

_VIX_SAFE_HAVEN = 22.0    # above → move to gold/liquid
_VIX_RISK_ON    = 14.0    # below → move to equity
_NAV_PREMIUM_THRESHOLD = 0.005  # 0.5% premium → sell ETF, buy components
_NAV_DISCOUNT_THRESHOLD = -0.005  # 0.5% discount → buy ETF


class ETFAgent(BaseAgent):
    """ETF arbitrage, sector rotation, and safe haven allocation."""

    AGENT_KEY = "etf"

    async def analyze(self, market_data: dict, sentiment_data: dict) -> list[AgentSignal]:
        signals: list[AgentSignal] = []
        symbol = market_data.get("symbol", "UNKNOWN")
        macro  = market_data.get("macro", {})
        quote  = market_data.get("quote", {})
        fund   = market_data.get("fundamentals", {})
        ind    = market_data.get("indicators", {})

        ltp            = float(quote.get("ltp", 0.0))
        atr            = float(ind.get("atr", 0.0))
        vix            = float(macro.get("india_vix", 16.0))
        nav            = float(fund.get("nav", ltp))               # true NAV from AMC
        nav_premium    = (ltp - nav) / nav if nav > 0 else 0.0
        sector_rank    = float(macro.get("sector_momentum_rank", 50.0))  # 0-100
        fii_streak     = int(macro.get("fii_buy_streak_days", 0))

        mult   = await self._calibration.get_sl_tp_multipliers()
        sizing = await self._calibration.get_position_sizing()
        sl_m   = mult.get("positional_sl_atr", 3.0)
        rr     = mult.get("target_risk_reward", 2.0)

        # ── NAV discount → buy ETF ────────────────────────────────────────────
        if nav_premium <= _NAV_DISCOUNT_THRESHOLD:
            signals.append(self._make_signal(
                symbol=symbol, action=Action.BUY, strength=0.65,
                reason=f"NAV discount: ETF trading {nav_premium:.2%} below NAV ({nav:.2f}) → arb BUY",
                strategy="nav_arb", entry=ltp,
                stop_loss=ltp - atr * sl_m, target=nav,
                position_size_pct=sizing.get("default_pct", 0.03),
                metadata={"nav": nav, "premium": nav_premium},
            ))

        # ── NAV premium → sell ETF ────────────────────────────────────────────
        elif nav_premium >= _NAV_PREMIUM_THRESHOLD:
            signals.append(self._make_signal(
                symbol=symbol, action=Action.SELL, strength=0.60,
                reason=f"NAV premium: ETF trading {nav_premium:.2%} above NAV ({nav:.2f}) → arb SELL",
                strategy="nav_arb", entry=ltp,
                stop_loss=ltp + atr * sl_m, target=nav,
                position_size_pct=sizing.get("low_conviction_pct", 0.01),
                metadata={"nav": nav, "premium": nav_premium},
            ))

        # ── Safe haven rotation: high VIX → gold/liquid ───────────────────────
        if vix >= _VIX_SAFE_HAVEN:
            if symbol == _EQUITY_ETF:
                signals.append(self._make_signal(
                    symbol=_EQUITY_ETF, action=Action.SELL, strength=0.70,
                    reason=f"VIX={vix:.1f} ≥ {_VIX_SAFE_HAVEN}: rotate out of equity ETF",
                    strategy="safe_haven_rotation", entry=ltp, stop_loss=0.0, target=0.0,
                    position_size_pct=sizing.get("default_pct", 0.03),
                ))
            if symbol in (_GOLD_ETF, _LIQUID_ETF):
                signals.append(self._make_signal(
                    symbol=symbol, action=Action.BUY, strength=0.70,
                    reason=f"VIX={vix:.1f} ≥ {_VIX_SAFE_HAVEN}: rotate into {symbol}",
                    strategy="safe_haven_rotation", entry=ltp,
                    stop_loss=ltp - atr * sl_m, target=ltp + atr * sl_m * rr,
                    position_size_pct=sizing.get("default_pct", 0.03),
                ))

        # ── Risk-on rotation: low VIX → equity ───────────────────────────────
        elif vix < _VIX_RISK_ON and symbol == _EQUITY_ETF:
            strength = min(1.0, 0.55 + (_VIX_RISK_ON - vix) / 5 * 0.20
                        + (0.10 if fii_streak >= 3 else 0))
            signals.append(self._make_signal(
                symbol=symbol, action=Action.BUY, strength=round(strength, 3),
                reason=(
                    f"Low volatility risk-on: VIX={vix:.1f} < {_VIX_RISK_ON}, "
                    f"FII streak={fii_streak}d → add equity ETF"
                ),
                strategy="risk_on_equity", entry=ltp,
                stop_loss=ltp - atr * sl_m, target=ltp + atr * sl_m * rr,
                position_size_pct=sizing.get("default_pct", 0.03),
            ))

        # ── Sector momentum rotation ──────────────────────────────────────────
        if sector_rank >= 80 and symbol in (_BANK_ETF, _IT_ETF):
            signals.append(self._make_signal(
                symbol=symbol, action=Action.BUY, strength=0.65,
                reason=f"Sector momentum: {symbol} rank={sector_rank:.0f}th pct → sector BUY",
                strategy="sector_rotation", entry=ltp,
                stop_loss=ltp - atr * sl_m, target=ltp + atr * sl_m * rr,
                position_size_pct=sizing.get("default_pct", 0.03),
                metadata={"sector_rank": sector_rank},
            ))
        elif sector_rank <= 20 and symbol in (_BANK_ETF, _IT_ETF):
            signals.append(self._make_signal(
                symbol=symbol, action=Action.SELL, strength=0.60,
                reason=f"Sector rotation: {symbol} rank={sector_rank:.0f}th pct → sector SELL",
                strategy="sector_rotation", entry=ltp,
                stop_loss=ltp + atr * sl_m, target=ltp - atr * sl_m * rr,
                position_size_pct=sizing.get("low_conviction_pct", 0.01),
            ))

        log.debug("ETFAgent %s: %d signal(s)", symbol, len(signals))
        return signals
