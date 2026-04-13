"""
NEXUS-II — MiroFish Report Extractor
Synthesizes simulation results into human-readable insights and conviction scores.

Takes SimulationResult output and generates:
  • Textual narrative explaining consensus
  • Conviction metrics (0–1 scores per symbol/sector)
  • Risk alerts (if significant divergence or tail risk signals)
  • Recommendations for Monday trading based on emergent consensus

Used by WeeklyReportGenerator for the Friday "weekend simulation" section.

Usage:
    extractor = ReportExtractor()
    report = extractor.extract(simulation_result)
    # report = {narrative, conviction_by_sector, alerts, recommendation}
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

from core.mirofish.simulation_runner import Direction, SimulationResult

log = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")


# ── Conviction score ─────────────────────────────────────────────────────────

@dataclass
class ConvictionScore:
    """Conviction metric for a symbol/sector."""
    target: str  # symbol or sector name
    direction: Direction
    conviction: float  # 0–1, higher = more confident
    bullish_agents: int
    bearish_agents: int
    neutral_agents: int
    avg_sizing: float
    risk_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "direction": self.direction.value,
            "conviction": round(self.conviction, 3),
            "bullish_agents": self.bullish_agents,
            "bearish_agents": self.bearish_agents,
            "neutral_agents": self.neutral_agents,
            "avg_sizing": round(self.avg_sizing, 4),
            "risk_flags": self.risk_flags,
        }


# ── Full report ───────────────────────────────────────────────────────────────

@dataclass
class MiroFishReport:
    """Complete weekend simulation report."""
    scenario_id: str
    simulation_ts: datetime
    macro_seeds: List[str]
    narrative: str
    conviction_scores: List[ConvictionScore] = field(default_factory=list)
    overall_direction: Direction = Direction.HOLD
    overall_conviction: float = 0.0
    alerts: List[str] = field(default_factory=list)
    recommendation: str = ""

    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "simulation_ts": self.simulation_ts.isoformat(),
            "macro_seeds": self.macro_seeds,
            "narrative": self.narrative,
            "conviction_scores": [c.to_dict() for c in self.conviction_scores],
            "overall_direction": self.overall_direction.value,
            "overall_conviction": round(self.overall_conviction, 3),
            "alerts": self.alerts,
            "recommendation": self.recommendation,
        }


# ── Report Extractor ─────────────────────────────────────────────────────────

class ReportExtractor:
    """
    Synthesizes SimulationResult into actionable insights.
    """

    async def extract(
        self,
        result: SimulationResult,
        macro_seeds: Optional[List[str]] = None,
    ) -> MiroFishReport:
        """
        Extract a human-readable report from simulation results.

        Parameters
        ----------
        result       : SimulationResult from SimulationRunner.run()
        macro_seeds  : List of macro seeds applied (for context).

        Returns
        -------
        MiroFishReport with narrative, conviction scores, and alerts.
        """
        seeds = macro_seeds or []

        # Extract conviction scores per symbol
        conviction_scores = self._extract_conviction_scores(result)

        # Generate narrative
        narrative = self._generate_narrative(
            result, conviction_scores, seeds
        )

        # Overall direction / conviction
        overall_dir, overall_conv = self._compute_overall(conviction_scores)

        # Alerts
        alerts = self._identify_alerts(result, conviction_scores)

        # Recommendation
        recommendation = self._generate_recommendation(
            overall_dir, overall_conv, alerts
        )

        report = MiroFishReport(
            scenario_id=result.scenario_id,
            simulation_ts=result.execution_ts,
            macro_seeds=seeds,
            narrative=narrative,
            conviction_scores=conviction_scores,
            overall_direction=overall_dir,
            overall_conviction=overall_conv,
            alerts=alerts,
            recommendation=recommendation,
        )

        log.info(
            "Extracted MiroFish report for %s: %s conviction=%.1f%%",
            result.scenario_id, overall_dir.value, overall_conv * 100,
        )
        return report

    # ── Conviction extraction ─────────────────────────────────────────────

    @staticmethod
    def _extract_conviction_scores(result: SimulationResult) -> List[ConvictionScore]:
        """Extract per-symbol conviction from simulation rounds."""
        scores = []

        # Group decisions by symbol
        by_symbol: Dict[str, list] = {}
        for round_result in result.rounds:
            sym = round_result.symbol
            if sym not in by_symbol:
                by_symbol[sym] = []
            by_symbol[sym].extend(round_result.decisions)

        for symbol, decisions in by_symbol.items():
            if not decisions:
                continue

            buy_cnt = sum(1 for d in decisions if d.direction == Direction.BUY)
            sell_cnt = sum(1 for d in decisions if d.direction == Direction.SELL)
            hold_cnt = len(decisions) - buy_cnt - sell_cnt
            total = len(decisions)

            # Consensus direction
            if buy_cnt > sell_cnt:
                direction = Direction.BUY
                conviction = (buy_cnt - sell_cnt) / total
            elif sell_cnt > buy_cnt:
                direction = Direction.SELL
                conviction = (sell_cnt - buy_cnt) / total
            else:
                direction = Direction.HOLD
                conviction = 0.3

            avg_sizing = sum(d.position_size_pct for d in decisions) / total

            score = ConvictionScore(
                target=symbol,
                direction=direction,
                conviction=min(conviction, 1.0),
                bullish_agents=buy_cnt,
                bearish_agents=sell_cnt,
                neutral_agents=hold_cnt,
                avg_sizing=avg_sizing,
            )
            scores.append(score)

        return scores

    # ── Narrative generation ──────────────────────────────────────────────

    @staticmethod
    def _generate_narrative(
        result: SimulationResult,
        conviction_scores: List[ConvictionScore],
        macro_seeds: List[str],
    ) -> str:
        """Generate human-readable summary of simulation."""
        lines = []

        lines.append("## MiroFish Weekend Simulation Report")
        lines.append(f"**Scenario**: {result.scenario_id}")
        lines.append(f"**Agents**: {result.num_agents} personalities across {len(result.archetype_breakdown)} archetypes")

        if macro_seeds:
            lines.append(f"**Macro Seeds**: {', '.join(macro_seeds)}")

        lines.append("")
        lines.append("### Consensus Overview")

        for score in conviction_scores:
            pct = score.conviction * 100
            label = f"{'🟢' if score.direction == Direction.BUY else '🔴' if score.direction == Direction.SELL else '⚪'}"
            lines.append(
                f"{label} **{score.target}**: {score.direction.value} "
                f"({pct:.0f}% conviction) — {score.bullish_agents}B / "
                f"{score.bearish_agents}S / {score.neutral_agents}H"
            )

        lines.append("")
        lines.append("### Agent Archetypes")
        for archetype, pct in result.archetype_breakdown.items():
            lines.append(f"- {archetype}: {pct:.1%}")

        lines.append("")
        lines.append(f"**Risk Appetite**: {result.risk_appetite:.1%}")
        lines.append(f"**Average Sizing**: ~{sum(s.avg_sizing for s in conviction_scores) / len(conviction_scores) * 100:.2f}% per trade")

        return "\n".join(lines)

    @staticmethod
    def _compute_overall(
        conviction_scores: List[ConvictionScore],
    ) -> tuple:
        """Compute overall consensus direction and conviction."""
        if not conviction_scores:
            return Direction.HOLD, 0.0

        buy_scores = [s for s in conviction_scores if s.direction == Direction.BUY]
        sell_scores = [s for s in conviction_scores if s.direction == Direction.SELL]

        if len(buy_scores) > len(sell_scores):
            direction = Direction.BUY
            conviction = sum(s.conviction for s in buy_scores) / len(conviction_scores)
        elif len(sell_scores) > len(buy_scores):
            direction = Direction.SELL
            conviction = sum(s.conviction for s in sell_scores) / len(conviction_scores)
        else:
            direction = Direction.HOLD
            conviction = 0.3

        return direction, min(conviction, 1.0)

    # ── Alert generation ─────────────────────────────────────────────────

    @staticmethod
    def _identify_alerts(
        result: SimulationResult,
        conviction_scores: List[ConvictionScore],
    ) -> List[str]:
        """Generate risk alerts based on simulation output."""
        alerts = []

        # High divergence alert
        for score in conviction_scores:
            if abs(score.bullish_agents - score.bearish_agents) < 10:
                alerts.append(
                    f"⚠️ {score.target}: Split decision ({score.bullish_agents}B vs {score.bearish_agents}S) — "
                    f"low conviction, higher risk."
                )

        # Very few agents agreeing
        for score in conviction_scores:
            if score.conviction < 0.3:
                alerts.append(
                    f"⚠️ {score.target}: Weak consensus ({score.conviction:.0%} conviction) — "
                    f"consider tighter risk controls."
                )

        return alerts

    @staticmethod
    def _generate_recommendation(
        overall_dir: Direction,
        overall_conv: float,
        alerts: List[str],
    ) -> str:
        """Generate trading recommendation for Monday."""
        if overall_dir == Direction.BUY:
            if overall_conv > 0.6:
                return (
                    "🟢 **RECOMMENDED: Increase Longs** — Strong bullish consensus across agents. "
                    "Consider increasing position sizes for growth sectors. Monitor RBI policy signals Monday."
                )
            else:
                return (
                    "🟡 **CAUTIOUS LONGS** — Moderate bullish lean. Maintain defensive stops. "
                    "Avoid over-sizing; risk/reward is unclear."
                )
        elif overall_dir == Direction.SELL:
            if overall_conv > 0.6:
                return (
                    "🔴 **RECOMMENDED: Increase Hedges** — Strong bearish consensus. "
                    "Reduce long exposure, prepare for downside. Consider Put spreads in volatile sectors."
                )
            else:
                return (
                    "🟡 **CAUTIOUS SHORTS** — Moderate bearish lean. Avoid aggressive short bets. "
                    "Prefer neutral or defensive positioning."
                )
        else:
            return (
                "⚪ **NEUTRAL / HOLD** — No clear consensus from agent swarm. "
                "Stick to mean reversion strategies and single-stock ideas. Avoid macro bets."
            )
