"""
NEXUS-II - DebateArena (Layer 4, Tier 3)
Structured 2-3 round Bull vs Bear debate with Risk Researcher guardrails.
Claude Sonnet synthesises the final verdict (NOT Opus per system design).

Protocol:
  Round 1: Bull presents thesis -> Bear critiques
  Round 2: Bear presents counter-thesis -> Bull rebuts
  Round 3 (optional): If |conviction_split| > 0.30
  Synthesis: Sonnet reads full transcript -> direction, conviction, sizing
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
from core.agents.researchers.bear_researcher import BearThesis
from core.agents.researchers.bull_researcher import BullThesis
from core.agents.researchers.risk_researcher import RiskAssessment

if TYPE_CHECKING:
    from core.calibration.weight_calibration_agent import WeightCalibrationAgent

log = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")

_CONVICTION_SPLIT_THRESHOLD = 0.30

_SYNTHESIS_SYSTEM = "\n".join([
    "You are the Debate Synthesiser for NEXUS-II, an Indian market trading system.",
    "Read the Bull vs Bear transcript and Risk guardrails. Produce the final calibrated verdict.",
    "Rules:",
    "1. Weigh argument quality, not just conviction numbers.",
    "2. If risk_rating=EXTREME -> direction must be HOLD.",
    "3. Conviction: 0.5=close debate, 0.9=one side won decisively.",
    "4. position_size_pct must be <= max allowed size.",
    "Return ONLY valid JSON (no markdown):",
    '{"direction":"BUY|SELL|HOLD","conviction":0.0,"position_size_pct":0.03,',
    '"winner":"BULL|BEAR|DRAW","decisive_argument":"...","summary":"...","trade_rationale":"..."}',
])

_REBUTTAL_SYSTEM = (
    "You are in a structured trading debate. Rebut the opposing argument concisely (2-3 sentences). "
    'Return ONLY JSON: {"rebuttal": "your rebuttal here"}'
)


@dataclass
class DebateVerdict:
    """Final output of the Debate Arena, passed to PortfolioManager."""
    symbol:             str
    direction:          str       # BUY / SELL / HOLD
    conviction:         float     # 0.0-1.0
    position_size_pct:  float
    winner:             str       # BULL / BEAR / DRAW
    decisive_argument:  str
    summary:            str
    trade_rationale:    str
    debate_transcript:  str
    rounds_run:         int
    timestamp:          datetime = field(default_factory=lambda: datetime.now(IST))


class DebateArena:
    """
    Layer 4: Structured Bull vs Bear debate with Claude Sonnet synthesis.
    Uses Claude Sonnet (NOT Opus) per system design v2.1.
    """

    def __init__(
        self,
        calibration_agent: "WeightCalibrationAgent",
        claude_client: anthropic.AsyncAnthropic,
    ) -> None:
        self._calibration = calibration_agent
        self._claude = claude_client

    async def run(
        self,
        symbol: str,
        bull: BullThesis,
        bear: BearThesis,
        risk: RiskAssessment,
        analyst_reports: list[AnalystReport],
    ) -> DebateVerdict:
        """Run the full debate and return a synthesised verdict."""

        # Hard override: EXTREME risk -> instant HOLD
        if risk.risk_rating == "EXTREME" or not risk.risk_reward_acceptable:
            log.info("DebateArena: EXTREME risk for %s -> instant HOLD", symbol)
            return DebateVerdict(
                symbol=symbol, direction="HOLD", conviction=0.0,
                position_size_pct=0.0, winner="BEAR",
                decisive_argument="Risk Researcher flagged EXTREME risk / R:R unacceptable",
                summary=risk.summary, trade_rationale="Risk override - no trade.",
                debate_transcript="[BYPASSED - EXTREME risk]", rounds_run=0,
            )

        sizing   = await self._calibration.get_position_sizing()
        max_size = min(risk.max_position_recommendation_pct, sizing.get("max_pct", 0.05))
        parts: list[str] = []

        # Round 1
        parts.append(
            "=== ROUND 1 ===\n"
            f"BULL THESIS: {bull.thesis}\n\n"
            f"BEAR CRITIQUE: {bear.counter_thesis}"
        )

        # Round 2 - rebuttals
        bull_reb = await self._get_rebuttal("BULL", bear.counter_thesis, bear.strongest_objections)
        bear_reb = await self._get_rebuttal("BEAR", bull.thesis, bull.strongest_arguments)
        parts.append(
            "\n=== ROUND 2 ===\n"
            f"BULL REBUTTAL: {bull_reb}\n\n"
            f"BEAR REBUTTAL: {bear_reb}"
        )
        rounds_run = 2

        # Round 3 optional tiebreaker
        if abs(bull.conviction - bear.conviction) > _CONVICTION_SPLIT_THRESHOLD:
            tb = await self._tiebreaker(symbol, bull, bear, risk, bull_reb, bear_reb)
            parts.append(f"\n=== ROUND 3 (TIEBREAKER) ===\n{tb}")
            rounds_run = 3

        transcript = "\n\n".join(parts)
        risk_context = (
            f"Risk Rating: {risk.risk_rating} | R:R Acceptable: {risk.risk_reward_acceptable}\n"
            f"Key Risks: {'; '.join(risk.key_risks[:3])}\n"
            f"Guardrails: {'; '.join(risk.guardrails[:2])}\n"
            f"Max Position: {risk.max_position_recommendation_pct:.1%}"
        )

        verdict = await self._synthesise(symbol, transcript, risk_context, max_size)
        verdict.rounds_run = rounds_run
        verdict.debate_transcript = transcript

        log.info(
            "DebateArena %s -> %s (conviction=%.2f, winner=%s, rounds=%d)",
            symbol, verdict.direction, verdict.conviction, verdict.winner, rounds_run,
        )
        return verdict

    async def _get_rebuttal(self, side: str, opposing: str, arguments: list[str]) -> str:
        args_str = ", ".join(arguments[:3])
        prompt = f"You are the {side} side. Rebut:\nOpposing thesis: {opposing}\nArguments: {args_str}"
        try:
            resp = await self._claude.messages.create(
                model=LLM_MAIN, max_tokens=256, system=_REBUTTAL_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
            if raw.startswith("```"):
                raw = "\n".join(l for l in raw.splitlines() if not l.startswith("```")).strip()
            return json.loads(raw).get("rebuttal", raw)
        except Exception as exc:
            log.warning("Rebuttal failed (%s): %s", side, exc)
            return f"[{side} rebuttal unavailable]"

    async def _tiebreaker(
        self, symbol: str,
        bull: BullThesis, bear: BearThesis, risk: RiskAssessment,
        bull_reb: str, bear_reb: str,
    ) -> str:
        prompt = (
            f"Symbol: {symbol}\n"
            f"Bull conviction: {bull.conviction:.2f} | Bear conviction: {bear.conviction:.2f}\n"
            f"Bull rebuttal: {bull_reb}\nBear rebuttal: {bear_reb}\n"
            f"Risk: {risk.risk_rating} | Tail risk: {risk.tail_risk_scenario}\n"
            "What is the single most decisive factor? Answer in 2-3 sentences."
        )
        try:
            resp = await self._claude.messages.create(
                model=LLM_MAIN, max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text.strip()
        except Exception as exc:
            return f"[Tiebreaker unavailable: {exc}]"

    async def _synthesise(
        self, symbol: str, transcript: str, risk_context: str, max_size: float,
    ) -> DebateVerdict:
        prompt = (
            f"Symbol: {symbol}\n\n"
            f"DEBATE TRANSCRIPT:\n{transcript}\n\n"
            f"RISK GUARDRAILS:\n{risk_context}\n"
            f"Max allowed position size: {max_size:.1%}\n\n"
            "Provide the final verdict JSON."
        )
        try:
            resp = await self._claude.messages.create(
                model=LLM_MAIN, max_tokens=512, system=_SYNTHESIS_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
            if raw.startswith("```"):
                raw = "\n".join(l for l in raw.splitlines() if not l.startswith("```")).strip()
            data = json.loads(raw)
        except Exception as exc:
            log.error("DebateArena synthesis failed for %s: %s", symbol, exc)
            data = {
                "direction": "HOLD", "conviction": 0.0, "position_size_pct": 0.0,
                "winner": "DRAW", "decisive_argument": "Synthesis error",
                "summary": str(exc), "trade_rationale": "No trade - synthesis failed.",
            }

        return DebateVerdict(
            symbol=symbol,
            direction=data.get("direction", "HOLD").upper(),
            conviction=float(data.get("conviction", 0.0)),
            position_size_pct=min(float(data.get("position_size_pct", 0.03)), max_size),
            winner=data.get("winner", "DRAW").upper(),
            decisive_argument=data.get("decisive_argument", ""),
            summary=data.get("summary", ""),
            trade_rationale=data.get("trade_rationale", ""),
            debate_transcript="",  # filled by caller
            rounds_run=0,
        )
