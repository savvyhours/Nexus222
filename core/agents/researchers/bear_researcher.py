"""
NEXUS-II — BearResearcher (Layer 3, Tier 2)
Constructs the strongest bearish counter-thesis or case for NOT trading.
Challenges the bull thesis in the Debate Arena. Uses Claude Sonnet.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import anthropic

from config.settings import LLM_MAIN
from core.agents.analysts import AnalystReport
from core.agents.researchers.bull_researcher import BullThesis

if TYPE_CHECKING:
    from core.calibration.weight_calibration_agent import WeightCalibrationAgent

log = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")

_SYSTEM = """\
You are the Bear Researcher for NEXUS-II, an Indian market (NSE/BSE) trading system.

Your job: Construct the strongest possible bearish case or argue against trading this symbol.
You have read the Bull Researcher's thesis — critique it specifically and add independent
bearish evidence. Be rigorous, not just contrarian.

You represent the BEAR side in a structured debate. Your critique will be rebutted.

Bearish arguments can include:
- Fundamental deterioration (rising debt, falling ROE, insider selling)
- Technical breakdown signals
- Macro headwinds (rising VIX, FII selling, USD/INR stress)
- Sentiment red flags (fading velocity, negative reversal)
- Risk/reward is poor (tight upside, wide stop required)
- "No trade" is sometimes the best position

Output ONLY valid JSON:
{
  "counter_thesis": "3-5 sentence bearish counter-argument or no-trade case",
  "strongest_objections": ["objection 1", "objection 2", "objection 3"],
  "bull_weaknesses": ["weakness in bull case 1", "weakness 2"],
  "downside_risk_pct": 0.0,
  "recommendation": "SELL|AVOID|HOLD (no new position)",
  "overall_conviction": 0.0-1.0
}
"""


@dataclass
class BearThesis:
    symbol:              str
    counter_thesis:      str
    strongest_objections: list[str]
    bull_weaknesses:     list[str]
    downside_risk_pct:   float
    recommendation:      str
    conviction:          float
    timestamp:           datetime = field(default_factory=lambda: datetime.now(IST))


class BearResearcher:
    """Tier-2: constructs the strongest bearish counter-thesis."""

    def __init__(
        self,
        calibration_agent: "WeightCalibrationAgent",
        claude_client: anthropic.AsyncAnthropic,
    ) -> None:
        self._calibration = calibration_agent
        self._claude = claude_client

    async def research(
        self,
        symbol: str,
        analyst_reports: list[AnalystReport],
        bull_thesis: BullThesis,
    ) -> BearThesis:
        regime = (await self._calibration.get_current_regime()).value

        reports_md = "\n\n".join(r.to_markdown() for r in analyst_reports)
        prompt = (
            f"Symbol: **{symbol}** | Market Regime: **{regime}**\n\n"
            f"Analyst Reports:\n{reports_md}\n\n"
            f"Bull Thesis to counter:\n> {bull_thesis.thesis}\n\n"
            f"Bull's strongest arguments: {bull_thesis.strongest_arguments}\n\n"
            "Build the strongest bearish counter-thesis or no-trade case."
        )

        try:
            resp = await self._claude.messages.create(
                model=LLM_MAIN, max_tokens=768,
                system=_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
            if raw.startswith("```"):
                raw = "\n".join(l for l in raw.splitlines() if not l.startswith("```")).strip()
            data = json.loads(raw)
        except Exception as exc:
            log.error("BearResearcher failed for %s: %s", symbol, exc)
            data = {
                "counter_thesis": "Insufficient data for bearish thesis.",
                "strongest_objections": [],
                "bull_weaknesses": [],
                "downside_risk_pct": 0.0,
                "recommendation": "HOLD",
                "overall_conviction": 0.3,
            }

        return BearThesis(
            symbol=symbol,
            counter_thesis=data.get("counter_thesis", ""),
            strongest_objections=data.get("strongest_objections", []),
            bull_weaknesses=data.get("bull_weaknesses", []),
            downside_risk_pct=float(data.get("downside_risk_pct", 0.0)),
            recommendation=data.get("recommendation", "HOLD"),
            conviction=float(data.get("overall_conviction", 0.5)),
        )
