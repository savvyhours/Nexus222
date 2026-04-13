"""
NEXUS-II — BullResearcher (Layer 3, Tier 2)
Constructs the strongest possible bullish investment thesis for a symbol
by synthesising all four Tier-1 analyst reports. Uses Claude Sonnet.
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

if TYPE_CHECKING:
    from core.calibration.weight_calibration_agent import WeightCalibrationAgent

log = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")

_SYSTEM = """\
You are the Bull Researcher for NEXUS-II, an Indian market (NSE/BSE) trading system.

Your job: Synthesise all analyst reports and construct the STRONGEST possible bullish case
for the proposed trade. Be rigorous — cherry-pick the strongest bullish evidence from each
analyst. Acknowledge weaknesses only to show you've considered them.

You represent the BULL side in a structured debate. Your thesis will be challenged by the
Bear Researcher. Make your case compelling but factually grounded.

Output ONLY valid JSON:
{
  "thesis": "3-5 sentence bullish investment thesis",
  "strongest_arguments": ["arg 1", "arg 2", "arg 3"],
  "supporting_analysts": ["technical", "sentiment", ...],
  "upside_target_pct": 0.0,
  "risk_acknowledgment": "1 sentence acknowledging the main bear risk",
  "overall_conviction": 0.0-1.0
}
"""


@dataclass
class BullThesis:
    symbol:              str
    thesis:              str
    strongest_arguments: list[str]
    supporting_analysts: list[str]
    upside_target_pct:   float
    risk_acknowledgment: str
    conviction:          float
    timestamp:           datetime = field(default_factory=lambda: datetime.now(IST))


class BullResearcher:
    """Tier-2: constructs the strongest bullish thesis from analyst reports."""

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
    ) -> BullThesis:
        regime = (await self._calibration.get_current_regime()).value

        reports_md = "\n\n".join(r.to_markdown() for r in analyst_reports)
        prompt = (
            f"Symbol: **{symbol}** | Market Regime: **{regime}**\n\n"
            f"Analyst Reports:\n{reports_md}\n\n"
            "Build the strongest bullish thesis."
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
            log.error("BullResearcher failed for %s: %s", symbol, exc)
            data = {
                "thesis": "Insufficient data for bullish thesis.",
                "strongest_arguments": [],
                "supporting_analysts": [],
                "upside_target_pct": 0.0,
                "risk_acknowledgment": "N/A",
                "overall_conviction": 0.3,
            }

        return BullThesis(
            symbol=symbol,
            thesis=data.get("thesis", ""),
            strongest_arguments=data.get("strongest_arguments", []),
            supporting_analysts=data.get("supporting_analysts", []),
            upside_target_pct=float(data.get("upside_target_pct", 0.0)),
            risk_acknowledgment=data.get("risk_acknowledgment", ""),
            conviction=float(data.get("overall_conviction", 0.5)),
        )
