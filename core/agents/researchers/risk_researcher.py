"""
NEXUS-II — RiskResearcher (Layer 3, Tier 2)
Independent risk assessment that provides guardrails throughout the debate.
Does NOT take a directional view — focuses on what could go wrong and
whether the risk/reward is acceptable. Uses Claude Sonnet.
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
You are the Risk Researcher for NEXUS-II, an Indian market (NSE/BSE) trading system.

Your role is INDEPENDENT RISK ASSESSMENT — you are not bullish or bearish.
You evaluate: What can go wrong? Is the risk/reward acceptable? Are there portfolio-level
concerns (correlation, sector concentration, drawdown context)?

You provide risk guardrails that the Debate Arena and Portfolio Manager both respect.

Evaluate:
1. Market risk: VIX level, upcoming events (RBI, earnings, Budget)
2. Execution risk: liquidity, bid-ask spread, circuit breaker proximity
3. Model risk: are agent signals based on thin data or genuine confluence?
4. Portfolio risk: how does this trade interact with existing positions?
5. Tail risk: what's the worst plausible outcome in 5 days?
6. Risk/Reward: is the R:R ratio ≥ 1.5? If not, flag it.

Output ONLY valid JSON:
{
  "risk_rating": "LOW|MEDIUM|HIGH|EXTREME",
  "risk_reward_acceptable": true/false,
  "key_risks": ["risk 1", "risk 2", "risk 3"],
  "tail_risk_scenario": "1 sentence worst-case",
  "portfolio_concerns": ["concern 1"] or [],
  "guardrails": ["guardrail 1", "guardrail 2"],
  "max_position_recommendation_pct": 0.03,
  "summary": "2-3 sentence risk assessment"
}
"""


@dataclass
class RiskAssessment:
    symbol:                         str
    risk_rating:                    str   # LOW / MEDIUM / HIGH / EXTREME
    risk_reward_acceptable:         bool
    key_risks:                      list[str]
    tail_risk_scenario:             str
    portfolio_concerns:             list[str]
    guardrails:                     list[str]
    max_position_recommendation_pct: float
    summary:                        str
    timestamp:                      datetime = field(default_factory=lambda: datetime.now(IST))


class RiskResearcher:
    """Tier-2: independent risk assessment and portfolio guardrails."""

    def __init__(
        self,
        calibration_agent: "WeightCalibrationAgent",
        claude_client: anthropic.AsyncAnthropic,
    ) -> None:
        self._calibration = calibration_agent
        self._claude = claude_client

    async def assess(
        self,
        symbol: str,
        analyst_reports: list[AnalystReport],
        portfolio_context: dict | None = None,
    ) -> RiskAssessment:
        regime    = (await self._calibration.get_current_regime()).value
        rt        = await self._calibration.get_risk_thresholds()
        sizing    = await self._calibration.get_position_sizing()
        kill_switch = await self._calibration.is_kill_switch_active()

        reports_md = "\n\n".join(r.to_markdown() for r in analyst_reports)
        ctx = portfolio_context or {}
        ctx["regime"] = regime
        ctx["kill_switch_active"] = kill_switch
        ctx["risk_thresholds"] = rt

        prompt = (
            f"Risk assessment for **{symbol}**\n\n"
            f"Analyst Reports:\n{reports_md}\n\n"
            f"Portfolio Context:\n```json\n{json.dumps(ctx, indent=2, default=str)}\n```\n\n"
            "Provide independent risk guardrails."
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
            log.error("RiskResearcher failed for %s: %s", symbol, exc)
            data = {
                "risk_rating": "HIGH",
                "risk_reward_acceptable": False,
                "key_risks": ["LLM analysis failed"],
                "tail_risk_scenario": "Unknown",
                "portfolio_concerns": [],
                "guardrails": ["Use minimum position size"],
                "max_position_recommendation_pct": sizing.get("low_conviction_pct", 0.01),
                "summary": "Risk analysis unavailable — conservative defaults applied.",
            }

        # Hard override: kill switch → EXTREME
        if kill_switch:
            data["risk_rating"] = "EXTREME"
            data["risk_reward_acceptable"] = False

        return RiskAssessment(
            symbol=symbol,
            risk_rating=data.get("risk_rating", "HIGH"),
            risk_reward_acceptable=bool(data.get("risk_reward_acceptable", False)),
            key_risks=data.get("key_risks", []),
            tail_risk_scenario=data.get("tail_risk_scenario", ""),
            portfolio_concerns=data.get("portfolio_concerns", []),
            guardrails=data.get("guardrails", []),
            max_position_recommendation_pct=min(
                float(data.get("max_position_recommendation_pct", sizing.get("default_pct", 0.03))),
                float(rt.get("max_position_pct", 0.10)),
            ),
            summary=data.get("summary", ""),
        )
