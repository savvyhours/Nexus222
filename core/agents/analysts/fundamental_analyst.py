"""
NEXUS-II — Fundamental Analyst (Tier 1)

Analyses fundamental valuation and quality metrics using Claude Sonnet.
Data sources:
  - Screener.in: P/E, EPS growth, ROE, debt/equity, promoter holding
  - Qlib Alpha158 factor scores (from MultiMarketQlibPipeline)
  - Earnings surprise and guidance
  - FII/DII ownership trends

Focus: Is this stock fundamentally strong enough to justify a trade at current price?
       Filters out value traps and identifies high-quality stocks.

Output: AnalystReport with BUY / SELL / HOLD bias and 0–1 conviction.
"""
from __future__ import annotations

import json
import logging

from core.agents.analysts import AnalystReport, BaseAnalyst
from core.agents.base_agent import Action

log = logging.getLogger(__name__)

# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are the Fundamental Analyst for NEXUS-II, an AI trading system for Indian markets (NSE/BSE).

Your job: Given fundamental data for a single stock, assess its investment quality and produce a structured fundamental assessment.

== VALUATION FRAMEWORK ==

P/E RATIO (vs sector median):
- P/E < 0.7× sector median → potentially undervalued (BUY lean if quality is high)
- P/E 0.7–1.3× sector median → fairly valued (neutral)
- P/E > 1.3× sector median → premium valuation (needs strong growth to justify)
- Negative P/E → loss-making (bearish unless turnaround thesis)

EPS GROWTH (YoY):
- > 20% → strong growth (bullish)
- 10–20% → healthy growth (mildly bullish)
- 0–10% → tepid (neutral)
- Negative → earnings deterioration (bearish)

QUALITY METRICS:
- ROE > 15% → high-quality business (adds conviction)
- Debt/Equity < 0.5 → low financial risk (adds conviction)
- Promoter holding > 50% with increasing trend → alignment (bullish)
- FII holding trend increasing → institutional accumulation (bullish)

QLIB FACTOR SCORE:
- Factor score is a rank within NSE500 or market universe (0.0 = worst, 1.0 = best)
- Score > 0.7 → top decile (strong BUY lean)
- Score > 0.5 → above median (mildly bullish)
- Score < 0.3 → bottom tercile (bearish)

EARNINGS SURPRISE:
- Positive surprise (actual > estimate) → bullish near-term catalyst
- Negative surprise → bearish near-term catalyst
- First positive surprise after 2+ misses → high conviction reversal

== VALUE TRAP WARNINGS ==
Reduce conviction to < 0.30 if:
- High dividend yield BUT declining earnings
- Low P/E BUT rising debt and falling ROE
- Promoter pledging > 50% of holding

== OUTPUT FORMAT ==
Return ONLY valid JSON (no markdown):
{
  "signal": "BUY" | "SELL" | "HOLD",
  "conviction": <float 0.0–1.0>,
  "summary": "<2–3 sentences synthesising the fundamental picture>",
  "key_findings": ["<finding 1>", "<finding 2>", "<finding 3 optional>"],
  "quality_score": <float 0.0–1.0 — overall business quality assessment>,
  "value_trap_risk": <bool — true if value trap warning applies>
}
"""

# ── FundamentalAnalyst ────────────────────────────────────────────────────────

class FundamentalAnalyst(BaseAnalyst):
    """
    Tier-1 Fundamental Analyst.

    Expects market_data to contain a "fundamentals" sub-dict with:
        pe_ratio:           float — trailing P/E (None if loss-making)
        pe_sector_median:   float — sector median P/E
        eps_growth_yoy:     float — YoY EPS growth as decimal (e.g. 0.18 = 18%)
        eps_growth_qoq:     float — QoQ EPS growth
        roe:                float — Return on Equity (e.g. 0.22 = 22%)
        debt_equity:        float — debt-to-equity ratio
        promoter_holding:   float — promoter holding % (e.g. 0.55 = 55%)
        promoter_pledge_pct: float — pledged % of promoter holding
        fii_holding_pct:    float — FII ownership % of equity
        fii_holding_trend:  str  — "increasing" | "decreasing" | "stable"
        qlib_factor_score:  float — 0.0–1.0 rank in universe
        earnings_surprise_pct: float — % beat/miss vs consensus
        earnings_surprise_streak: int — consecutive beats (positive) or misses (negative)
        dividend_yield:     float — trailing dividend yield
        market_cap_cr:      float — market cap in ₹ crore
    """

    ANALYST_NAME = "fundamental"

    async def analyze(self, symbol: str, market_data: dict) -> AnalystReport:
        fund = market_data.get("fundamentals", {})

        raw_data = {
            "pe_ratio":              fund.get("pe_ratio"),
            "pe_sector_median":      fund.get("pe_sector_median"),
            "pe_vs_sector":          (
                round(fund["pe_ratio"] / fund["pe_sector_median"], 2)
                if fund.get("pe_ratio") and fund.get("pe_sector_median")
                else None
            ),
            "eps_growth_yoy":        fund.get("eps_growth_yoy"),
            "eps_growth_qoq":        fund.get("eps_growth_qoq"),
            "roe":                   fund.get("roe"),
            "debt_equity":           fund.get("debt_equity"),
            "promoter_holding":      fund.get("promoter_holding"),
            "promoter_pledge_pct":   fund.get("promoter_pledge_pct"),
            "fii_holding_pct":       fund.get("fii_holding_pct"),
            "fii_holding_trend":     fund.get("fii_holding_trend"),
            "qlib_factor_score":     fund.get("qlib_factor_score"),
            "earnings_surprise_pct": fund.get("earnings_surprise_pct"),
            "earnings_surprise_streak": fund.get("earnings_surprise_streak"),
            "dividend_yield":        fund.get("dividend_yield"),
            "market_cap_cr":         fund.get("market_cap_cr"),
        }

        user_content = (
            f"Analyse fundamentals for **{symbol}**.\n\n"
            f"```json\n{json.dumps(raw_data, indent=2, default=str)}\n```"
        )

        try:
            raw_text = await self._call_claude(_SYSTEM_PROMPT, user_content)
            parsed = json.loads(self._strip_fences(raw_text))

            signal = Action(parsed.get("signal", "HOLD").upper())
            conviction = float(parsed.get("conviction", 0.0))
            summary = parsed.get("summary", "")
            key_findings = parsed.get("key_findings", [])
            quality_score = float(parsed.get("quality_score", 0.5))
            value_trap_risk = bool(parsed.get("value_trap_risk", False))

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
                "quality_score": quality_score,
                "value_trap_risk": value_trap_risk,
            },
        )
