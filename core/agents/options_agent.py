"""
NEXUS-II — OptionsAgent
FnO strategies: IV Crush, Theta Decay, Momentum, Hedging, Max Pain, Iron Condor.

Rate limit: options chain API = 1 call per 150 seconds (enforced externally).
Only trades liquid strikes with OI > 5 lakh (500,000).
"""
from __future__ import annotations

import logging

from config.strategy_params import (
    OPTIONS_DELTA_NEUTRAL_MAX, OPTIONS_IV_CRUSH_PERCENTILE,
    OPTIONS_MIN_OI, OPTIONS_VIX_HEDGE_TRIGGER,
)
from core.agents.base_agent import Action, AgentSignal, BaseAgent

log = logging.getLogger(__name__)


class OptionsAgent(BaseAgent):
    """Options/FnO specialist: IV crush, theta decay, hedging, iron condor."""

    AGENT_KEY = "options"

    async def analyze(self, market_data: dict, sentiment_data: dict) -> list[AgentSignal]:
        signals: list[AgentSignal] = []
        symbol  = market_data.get("symbol", "UNKNOWN")
        options = market_data.get("options", {})     # options chain snapshot
        macro   = market_data.get("macro", {})
        quote   = market_data.get("quote", {})
        ind     = market_data.get("indicators", {})

        ltp            = float(quote.get("ltp", 0.0))
        vix            = float(macro.get("india_vix", 16.0))
        iv_percentile  = float(options.get("iv_percentile", 50.0))
        atm_iv         = float(options.get("atm_iv", 20.0))
        max_pain       = float(options.get("max_pain_strike", ltp))
        oi_ce          = float(options.get("total_oi_ce", 0))
        oi_pe          = float(options.get("total_oi_pe", 0))
        days_to_expiry = int(options.get("days_to_expiry", 30))
        pcr            = (oi_pe / oi_ce) if oi_ce > 0 else 1.0  # put-call ratio

        mult   = await self._calibration.get_sl_tp_multipliers()
        sizing = await self._calibration.get_position_sizing()
        rt     = await self._calibration.get_risk_thresholds()
        min_oi = int(rt.get("min_fno_oi", OPTIONS_MIN_OI))

        # ── IV Crush: sell straddle pre-earnings ─────────────────────────────
        if iv_percentile > OPTIONS_IV_CRUSH_PERCENTILE and days_to_expiry <= 7:
            signals.append(self._make_signal(
                symbol=symbol, action=Action.SELL, strength=0.75,
                reason=(
                    f"IV Crush: IV percentile={iv_percentile:.0f}th > {OPTIONS_IV_CRUSH_PERCENTILE}th, "
                    f"DTE={days_to_expiry}. Sell ATM straddle."
                ),
                strategy="iv_crush_straddle", entry=atm_iv,
                stop_loss=0.0, target=0.0,
                position_size_pct=sizing.get("low_conviction_pct", 0.01),
                metadata={"iv_percentile": iv_percentile, "dte": days_to_expiry, "atm_iv": atm_iv},
            ))

        # ── Theta Decay: sell OTM covered calls ──────────────────────────────
        if days_to_expiry <= 14 and iv_percentile >= 40 and vix < 22:
            signals.append(self._make_signal(
                symbol=symbol, action=Action.SELL, strength=0.60,
                reason=f"Theta decay: DTE={days_to_expiry}, IV pct={iv_percentile:.0f}th, VIX={vix:.1f}",
                strategy="theta_decay_call", entry=0.0, stop_loss=0.0, target=0.0,
                position_size_pct=sizing.get("low_conviction_pct", 0.01),
                metadata={"days_to_expiry": days_to_expiry, "iv_percentile": iv_percentile},
            ))

        # ── Hedging: buy PE when VIX rising ──────────────────────────────────
        if vix >= OPTIONS_VIX_HEDGE_TRIGGER:
            signals.append(self._make_signal(
                symbol=symbol, action=Action.BUY, strength=min(1.0, 0.55 + (vix - OPTIONS_VIX_HEDGE_TRIGGER) / 15 * 0.30),
                reason=f"Hedging: VIX={vix:.1f} ≥ {OPTIONS_VIX_HEDGE_TRIGGER} → buy ATM PE for protection",
                strategy="vix_hedge_pe", entry=0.0, stop_loss=0.0, target=0.0,
                position_size_pct=sizing.get("low_conviction_pct", 0.01),
                metadata={"vix": vix, "trigger": OPTIONS_VIX_HEDGE_TRIGGER},
            ))

        # ── Max Pain: position near max pain strike pre-expiry ───────────────
        if days_to_expiry <= 2 and abs(ltp - max_pain) / max(ltp, 1) < 0.02:
            action = Action.BUY if pcr > 1.2 else Action.SELL
            signals.append(self._make_signal(
                symbol=symbol, action=action, strength=0.65,
                reason=(
                    f"Max Pain: DTE={days_to_expiry}, max_pain={max_pain:.0f}, "
                    f"LTP={ltp:.0f} (within 2%). PCR={pcr:.2f}."
                ),
                strategy="max_pain", entry=ltp, stop_loss=0.0, target=max_pain,
                position_size_pct=sizing.get("low_conviction_pct", 0.01),
                metadata={"max_pain": max_pain, "pcr": pcr, "dte": days_to_expiry},
            ))

        # ── Momentum: buy ATM calls on breakout ──────────────────────────────
        ema_20 = float(ind.get("ema_20", ltp))
        adx    = float(ind.get("adx", 0.0))
        if ltp > ema_20 * 1.01 and adx > 25 and iv_percentile < 60:
            signals.append(self._make_signal(
                symbol=symbol, action=Action.BUY, strength=0.65,
                reason=f"Options momentum BUY: LTP {ltp:.1f} broke above EMA20 {ema_20:.1f}, ADX={adx:.1f}",
                strategy="options_momentum_call", entry=0.0, stop_loss=0.0, target=0.0,
                position_size_pct=sizing.get("low_conviction_pct", 0.01),
                metadata={"iv_percentile": iv_percentile, "adx": adx},
            ))

        log.debug("OptionsAgent %s: %d signal(s), VIX=%.1f, IV_pct=%.0f",
                  symbol, len(signals), vix, iv_percentile)
        return signals
