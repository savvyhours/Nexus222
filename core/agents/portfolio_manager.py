"""
NEXUS-II — PortfolioManager (Layer 5)
Claude Sonnet reviews each ConsensusSignal from the debate arena and decides:
  - APPROVE: trade goes to risk layer
  - REJECT: trade dropped
  - MODIFY: trade approved with adjusted size/SL/target

Also enforces portfolio-level constraints: diversification, correlation,
capital allocation across all approved trades.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional
from zoneinfo import ZoneInfo

import anthropic

from config.settings import LLM_MAIN
from core.agents.base_agent import Action

if TYPE_CHECKING:
    from core.agents.master_orchestrator import ConsensusSignal
    from core.calibration.weight_calibration_agent import WeightCalibrationAgent

log = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")


class PMDecision(str, Enum):
    APPROVE = "APPROVE"
    REJECT  = "REJECT"
    MODIFY  = "MODIFY"


@dataclass
class TradeProposal:
    """
    Output of PortfolioManager — approved/modified trade ready for risk layer.
    """
    symbol:            str
    action:            Action
    conviction:        float
    entry_price:       float
    stop_loss:         float
    target:            float
    position_size_pct: float
    decision:          PMDecision
    reasoning:         str
    timestamp:         datetime = field(default_factory=lambda: datetime.now(IST))
    metadata:          dict = field(default_factory=dict)


_PM_SYSTEM = """\
You are the Portfolio Manager for NEXUS-II, an Indian market trading system.

Your role (Layer 5): Review each trade proposal coming from the Debate Arena and decide
APPROVE / REJECT / MODIFY. You also enforce portfolio-level discipline.

== APPROVAL CRITERIA ==
APPROVE if:
  - Conviction ≥ 0.60
  - Proposal aligns with current market regime
  - Portfolio correlation with existing positions < 0.80
  - Sector not already at max exposure
  - Risk/reward ≥ 1.5

REJECT if:
  - Conviction < 0.50
  - Adds too much sector/correlation risk to existing portfolio
  - Market regime contradicts the trade direction
  - P&L context suggests we are near daily loss limit

MODIFY if:
  - Good trade but oversized → reduce position_size_pct
  - Good trade but SL too tight → widen stop
  - Conviction borderline (0.50–0.60) → reduce to low_conviction size

== OUTPUT FORMAT ==
Return ONLY valid JSON:
{
  "decision": "APPROVE|REJECT|MODIFY",
  "position_size_pct": 0.03,
  "stop_loss": 0.0,
  "target": 0.0,
  "reasoning": "1-2 sentence rationale"
}
"""


class PortfolioManager:
    """
    Layer 5: Claude Sonnet portfolio-level trade approver.

    Reviews each ConsensusSignal (after debate), checks portfolio-level
    constraints (correlation, sector exposure, capital), and emits
    TradeProposal objects for the risk layer.
    """

    def __init__(
        self,
        calibration_agent: "WeightCalibrationAgent",
        claude_client: anthropic.AsyncAnthropic,
    ) -> None:
        self._calibration = calibration_agent
        self._claude = claude_client
        self._open_positions: dict[str, dict] = {}   # symbol → position snapshot

    # ── Public API ────────────────────────────────────────────────────────────

    def update_positions(self, positions: dict[str, dict]) -> None:
        """Called by execution layer after each fill/exit to keep PM in sync."""
        self._open_positions = positions

    async def review(
        self,
        consensus: "ConsensusSignal",
        debate_summary: Optional[str] = None,
    ) -> Optional[TradeProposal]:
        """
        Review a single ConsensusSignal.
        Returns TradeProposal if approved/modified, None if rejected.
        """
        # Kill switch
        if await self._calibration.is_kill_switch_active():
            return None

        sizing     = await self._calibration.get_position_sizing()
        thresholds = await self._calibration.get_risk_thresholds()
        regime     = (await self._calibration.get_current_regime()).value

        # Quick pre-filter: very low conviction → auto-reject before LLM call
        if consensus.conviction < 0.45:
            log.debug("PM auto-reject %s: conviction=%.2f < 0.45", consensus.symbol, consensus.conviction)
            return None

        # Check sector/correlation constraints locally before calling LLM
        sector_ok = self._check_sector_exposure(consensus.symbol, thresholds)
        corr_ok   = self._check_correlation(consensus.symbol, thresholds)

        # Build proposal context for LLM
        best_signal = max(consensus.agent_signals, key=lambda s: s.strength, default=None)
        entry  = best_signal.entry      if best_signal else 0.0
        sl     = best_signal.stop_loss  if best_signal else 0.0
        target = best_signal.target     if best_signal else 0.0
        default_size = sizing.get(
            "high_conviction_pct" if consensus.conviction >= 0.75 else "default_pct", 0.03
        )

        context = {
            "symbol":         consensus.symbol,
            "action":         consensus.action.value,
            "conviction":     consensus.conviction,
            "weighted_vote":  consensus.weighted_vote,
            "regime":         regime,
            "entry_price":    entry,
            "stop_loss":      sl,
            "target":         target,
            "default_size_pct": default_size,
            "sector_ok":      sector_ok,
            "correlation_ok": corr_ok,
            "open_positions_count": len(self._open_positions),
            "debate_summary": debate_summary or "(no debate summary provided)",
            "agent_votes":    [
                {"agent": s.agent_name, "action": s.action.value, "strength": s.strength}
                for s in consensus.agent_signals[:5]
            ],
        }

        prompt = (
            "Trade proposal for review:\n"
            + json.dumps(context, indent=2)
            + "\n\nDecide: APPROVE, REJECT, or MODIFY."
        )

        try:
            resp = await self._claude.messages.create(
                model=LLM_MAIN, max_tokens=512,
                system=_PM_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
            if raw.startswith("```"):
                raw = "\n".join(l for l in raw.splitlines() if not l.startswith("```")).strip()
            data = json.loads(raw)

            decision_str = data.get("decision", "REJECT").upper()
            decision     = PMDecision(decision_str)

            if decision == PMDecision.REJECT:
                log.info("PM rejected %s: %s", consensus.symbol, data.get("reasoning", ""))
                return None

            # Enforce safety bounds on PM output
            size = min(
                float(data.get("position_size_pct", default_size)),
                thresholds.get("max_position_pct", 0.10),
            )

            proposal = TradeProposal(
                symbol=consensus.symbol,
                action=consensus.action,
                conviction=consensus.conviction,
                entry_price=float(data.get("entry_price", entry) or entry),
                stop_loss=float(data.get("stop_loss", sl) or sl),
                target=float(data.get("target", target) or target),
                position_size_pct=size,
                decision=decision,
                reasoning=data.get("reasoning", ""),
                metadata={"regime": regime, "adjudicated": consensus.adjudicated},
            )

            log.info(
                "PM %s %s: size=%.1f%%, conviction=%.2f — %s",
                decision.value, consensus.symbol, size * 100,
                consensus.conviction, proposal.reasoning[:80],
            )
            return proposal

        except Exception as exc:
            log.error("PM LLM call failed for %s: %s", consensus.symbol, exc)
            return None

    async def review_batch(
        self,
        signals: list["ConsensusSignal"],
        debate_summaries: Optional[dict[str, str]] = None,
    ) -> list[TradeProposal]:
        """Review multiple signals; respects portfolio capital constraints."""
        import asyncio
        debate_summaries = debate_summaries or {}
        tasks = [
            self.review(sig, debate_summaries.get(sig.symbol))
            for sig in signals
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        proposals = []
        for r in results:
            if isinstance(r, Exception):
                log.error("PM review failed: %s", r)
            elif r is not None:
                proposals.append(r)

        # Capital allocation check: cap total deployed at once
        proposals = self._allocate_capital(proposals)
        return proposals

    # ── Portfolio constraint helpers ──────────────────────────────────────────

    def _check_sector_exposure(self, symbol: str, thresholds: dict) -> bool:
        """True if adding this symbol won't breach max_sector_pct."""
        max_sector = thresholds.get("max_sector_pct", 0.35)
        # In a full implementation this maps symbol → sector and sums existing exposure.
        # Here we use a simple heuristic: allow if open positions < 8.
        return len(self._open_positions) < 8

    def _check_correlation(self, symbol: str, thresholds: dict) -> bool:
        """True if adding symbol doesn't exceed max correlation with existing positions."""
        # Correlation check placeholder — full version uses stored correlation matrix.
        return symbol not in self._open_positions

    def _allocate_capital(self, proposals: list[TradeProposal]) -> list[TradeProposal]:
        """
        Ensure total position_size_pct across all new proposals ≤ 25%
        (25% new deployment per cycle to leave margin buffer).
        Sort by conviction descending, take until budget exhausted.
        """
        proposals.sort(key=lambda p: p.conviction, reverse=True)
        budget = 0.25
        allocated = []
        for p in proposals:
            if p.position_size_pct <= budget:
                allocated.append(p)
                budget -= p.position_size_pct
            else:
                p.position_size_pct = budget
                if p.position_size_pct >= 0.01:
                    allocated.append(p)
                break
        return allocated
