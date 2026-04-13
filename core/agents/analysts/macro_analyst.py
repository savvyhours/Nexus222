"""
NEXUS-II — Macro / Flow Analyst (Tier 1)

Analyses macro and institutional flow data using Claude Sonnet.
Data sources:
  - India VIX — fear gauge, options pricing environment
  - FII / DII net flows (3-day and 5-day running totals)
  - RBI calendar (rate decisions, CRR, policy stance changes)
  - Crude oil (Brent) — cost pressure / BoP signal for India
  - USD/INR — FII repatriation risk, IT sector signal
  - Sector momentum (NSE sector indices 5-day returns)

Macro signals that feed sub-agents:
  - FII buying 3 consecutive days → NIFTY BUY via MacroAgent
  - VIX > 22 → reduce all positions + GOLDBEES hedge recommendation
  - VIX < 14 → increase equity allocation
  - USD/INR > 85 → sell IT sector exposure

Output: AnalystReport with BUY / SELL / HOLD bias and 0–1 conviction.
         signal reflects the broad equity market direction, not a single stock.
         Researchers and sub-agents apply this context to individual symbols.
"""
from __future__ import annotations

import json
import logging

from core.agents.analysts import AnalystReport, BaseAnalyst
from core.agents.base_agent import Action

log = logging.getLogger(__name__)

# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are the Macro/Flow Analyst for NEXUS-II, an AI trading system for Indian markets (NSE/BSE).

Your job: Given current macro and institutional flow data, assess the broad market environment and produce a directional macro assessment.

== MACRO FRAMEWORK ==

INDIA VIX:
- VIX < 14: Low-volatility / complacency regime → equity-friendly, increase exposure
- VIX 14–20: Normal regime → base case
- VIX 20–22: Elevated fear → defensive posture, tighten stops
- VIX 22–28: High volatility → reduce position sizes 50%, hedge with GOLDBEES
- VIX > 28: CRISIS — system halted, exits only (kill switch engaged)

FII / DII FLOWS:
- FII net buying 3+ consecutive days → strong bullish signal for Nifty
- FII net selling 3+ consecutive days → bearish overhang, reduce longs
- DII absorbing FII selling (DII net buying while FII selling) → floor support, less bearish
- FII > ₹2,000cr net buying in single day → very bullish
- FII < −₹2,000cr net selling in single day → very bearish

USD / INR:
- USD/INR < 83: Favourable for FII inflows, neutral-bullish
- USD/INR 83–85: Elevated, watch IT sector
- USD/INR > 85: Sell IT sector, potential FII outflow concern
- Rapid INR depreciation (> 1% in week): risk-off signal

CRUDE OIL (Brent, USD/barrel):
- Crude < $70: Positive for India macro (lower import bill), bullish
- Crude $70–85: Neutral
- Crude > $85: Pressure on CAD and inflation, mildly bearish
- Rapid crude spike (> 5% week): Risk-off signal, sector-specific (OMCs, airlines)

SECTOR MOMENTUM (5-day NSE sector index returns):
- Any sector with 5d return > 3% → momentum trade opportunity in that sector
- Any sector with 5d return < −3% → avoid / short opportunities

RBI CALENDAR:
- Rate cut: Bullish for equities, especially rate-sensitive sectors (NBFCs, real estate)
- Rate hike: Bearish for equities, positive for INR
- Policy meeting this week: Raise uncertainty discount (reduce conviction)
- CRR cut: Liquidity positive, bullish banks

== SIGNAL RULES ==
- The macro signal represents the equity market backdrop, not a single stock
- Combine VIX + FII + Currency + Crude into one net direction
- If 3 of 4 factors align → high conviction; if split → HOLD with low conviction
- Always note any upcoming RBI/Budget events as risk factors

== OUTPUT FORMAT ==
Return ONLY valid JSON (no markdown):
{
  "signal": "BUY" | "SELL" | "HOLD",
  "conviction": <float 0.0–1.0>,
  "summary": "<2–3 sentences synthesising the macro picture>",
  "key_findings": ["<finding 1>", "<finding 2>", "<finding 3 optional>"],
  "regime_context": "<one of: RISK_ON | RISK_OFF | NEUTRAL>",
  "sector_calls": {"<sector>": "overweight|underweight|neutral", ...},
  "hedging_recommended": <bool — true if GOLDBEES/defensive allocation recommended>,
  "upcoming_risk_events": ["<event 1>", ...]
}
"""

# ── MacroAnalyst ──────────────────────────────────────────────────────────────

class MacroAnalyst(BaseAnalyst):
    """
    Tier-1 Macro / Flow Analyst.

    Expects market_data to contain a "macro" sub-dict with:
        india_vix:          float — current India VIX value
        fii_net_today_cr:   float — FII net flow today in ₹ crore
        fii_net_3d_cr:      float — FII cumulative net flow last 3 days
        fii_net_5d_cr:      float — FII cumulative net flow last 5 days
        fii_consecutive_days: int — consecutive buying (+) or selling (-) days
        dii_net_today_cr:   float — DII net flow today in ₹ crore
        dii_net_3d_cr:      float — DII cumulative net flow last 3 days
        usd_inr:            float — current USD/INR rate
        usd_inr_1w_change:  float — 1-week change in USD/INR
        crude_brent:        float — Brent crude in USD/barrel
        crude_1w_change_pct: float — 1-week % change in crude
        rbi_events:         list[str] — upcoming RBI / Budget events (next 5 days)
        sector_momentum_5d: dict — {"IT": 0.021, "BANKING": -0.015, ...}
        nifty_5d_change_pct: float — Nifty 50 5-day return
    """

    ANALYST_NAME = "macro"

    async def analyze(self, symbol: str, market_data: dict) -> AnalystReport:
        macro = market_data.get("macro", {})

        raw_data = {
            "india_vix":            macro.get("india_vix"),
            "fii_net_today_cr":     macro.get("fii_net_today_cr"),
            "fii_net_3d_cr":        macro.get("fii_net_3d_cr"),
            "fii_net_5d_cr":        macro.get("fii_net_5d_cr"),
            "fii_consecutive_days": macro.get("fii_consecutive_days"),
            "dii_net_today_cr":     macro.get("dii_net_today_cr"),
            "dii_net_3d_cr":        macro.get("dii_net_3d_cr"),
            "usd_inr":              macro.get("usd_inr"),
            "usd_inr_1w_change":    macro.get("usd_inr_1w_change"),
            "crude_brent":          macro.get("crude_brent"),
            "crude_1w_change_pct":  macro.get("crude_1w_change_pct"),
            "rbi_events":           macro.get("rbi_events", []),
            "sector_momentum_5d":   macro.get("sector_momentum_5d", {}),
            "nifty_5d_change_pct":  macro.get("nifty_5d_change_pct"),
            "symbol_context":       symbol,
        }

        user_content = (
            f"Analyse the macro/flow environment. Symbol context: **{symbol}**.\n\n"
            f"```json\n{json.dumps(raw_data, indent=2, default=str)}\n```"
        )

        try:
            raw_text = await self._call_claude(_SYSTEM_PROMPT, user_content)
            parsed = json.loads(self._strip_fences(raw_text))

            signal = Action(parsed.get("signal", "HOLD").upper())
            conviction = float(parsed.get("conviction", 0.0))
            summary = parsed.get("summary", "")
            key_findings = parsed.get("key_findings", [])
            regime_context = parsed.get("regime_context", "NEUTRAL")
            sector_calls = parsed.get("sector_calls", {})
            hedging = bool(parsed.get("hedging_recommended", False))
            risk_events = parsed.get("upcoming_risk_events", [])

        except Exception as exc:
            return self._fallback_report(symbol, raw_data, str(exc))

        return AnalystReport(
            analyst=self.ANALYST_NAME,
            symbol=symbol,
            signal=signal,
            conviction=conviction,
            summary=summary,
            key_findings=key_findings,
            raw_data=raw_data,
            metadata={
                "regime_context": regime_context,
                "sector_calls": sector_calls,
                "hedging_recommended": hedging,
                "upcoming_risk_events": risk_events,
            },
        )
