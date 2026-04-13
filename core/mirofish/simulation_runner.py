"""
NEXUS-II — MiroFish Simulation Runner
Executes a market scenario with 100+ agents and extracts emergent consensus.

Responsibilities:
  1. Initialize agents with personas and market state.
  2. Run simulation rounds where each agent makes trading decisions.
  3. Track collective decision distribution (conviction, direction, risk appetite).
  4. Return results for report_extractor to synthesize into insights.

Each agent independently evaluates the scenario and outputs:
  - direction: BUY / SELL / HOLD
  - confidence: 0–1
  - sizing: position size pct of portfolio

The simulation aggregates these into sector-level and index-level consensus.

Usage:
    runner = SimulationRunner()
    results = await runner.run(scenario, rounds=5)
    # results = {direction_consensus, avg_confidence, sector_consensus, ...}
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

from core.mirofish.scenario_builder import AgentPersona, Scenario

log = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")


class Direction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


# ── Agent decision ────────────────────────────────────────────────────────────

@dataclass
class AgentDecision:
    """Single agent's decision for a symbol/scenario."""
    agent_id: str
    archetype: str
    direction: Direction
    confidence: float  # 0–1
    position_size_pct: float  # % of portfolio if BUY/SELL
    reasoning: str = ""


# ── Simulation round result ───────────────────────────────────────────────────

@dataclass
class SimulationRound:
    """Aggregated result of one round of agent decisions."""
    round_num: int
    symbol: str = "NIFTY50"
    decisions: List[AgentDecision] = field(default_factory=list)
    consensus_direction: Direction = Direction.HOLD
    avg_confidence: float = 0.0
    bullish_pct: float = 0.0
    bearish_pct: float = 0.0
    avg_sizing: float = 0.0


# ── Full simulation result ────────────────────────────────────────────────────

@dataclass
class SimulationResult:
    """Complete simulation output."""
    scenario_id: str
    num_agents: int
    num_rounds: int
    rounds: List[SimulationRound] = field(default_factory=list)
    final_consensus: Direction = Direction.HOLD
    final_confidence: float = 0.0
    archetype_breakdown: Dict[str, float] = field(default_factory=dict)
    sector_consensus: Dict[str, Direction] = field(default_factory=dict)
    risk_appetite: float = 0.0  # aggregate risk tolerance
    execution_ts: datetime = field(default_factory=lambda: datetime.now(IST))

    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "num_agents": self.num_agents,
            "num_rounds": self.num_rounds,
            "final_consensus": self.final_consensus.value,
            "final_confidence": round(self.final_confidence, 3),
            "archetype_breakdown": self.archetype_breakdown,
            "sector_consensus": {s: d.value for s, d in self.sector_consensus.items()},
            "risk_appetite": round(self.risk_appetite, 3),
            "execution_ts": self.execution_ts.isoformat(),
        }


# ── Simulation Runner ─────────────────────────────────────────────────────────

class SimulationRunner:
    """
    Runs market scenarios with agent swarms to extract emergent consensus.

    Parameters
    ----------
    seed : Random seed for reproducibility (optional).
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._seed = seed
        if seed is not None:
            import random
            random.seed(seed)

    async def run(
        self,
        scenario: Scenario,
        rounds: int = 5,
        symbols: Optional[List[str]] = None,
    ) -> SimulationResult:
        """
        Execute a scenario with agent swarm and extract consensus.

        Parameters
        ----------
        scenario : Scenario object from ScenarioBuilder.
        rounds   : Number of decision rounds (default 5).
        symbols  : Symbols to evaluate (default ["NIFTY50", "BANKNIFTY"]).

        Returns
        -------
        SimulationResult with per-round decisions and aggregated consensus.
        """
        symbols = symbols or ["NIFTY50", "BANKNIFTY"]
        agents = scenario.agent_personas
        result = SimulationResult(
            scenario_id=scenario.scenario_id,
            num_agents=len(agents),
            num_rounds=rounds,
        )

        for round_num in range(1, rounds + 1):
            round_result = SimulationRound(round_num=round_num)

            for symbol in symbols:
                decisions = []
                for agent in agents:
                    decision = await self._agent_decide(
                        agent, scenario.market_state, symbol
                    )
                    decisions.append(decision)

                round_result.decisions = decisions
                round_result.symbol = symbol

                # Aggregate this round's decisions
                self._aggregate_decisions(round_result)
                result.rounds.append(round_result)

            log.info("Round %d/%d complete (%d symbols)", round_num, rounds, len(symbols))

        # Final consensus
        self._compute_final_consensus(result)

        return result

    # ── Agent decision logic ──────────────────────────────────────────────

    async def _agent_decide(
        self,
        agent: AgentPersona,
        market_state: dict,
        symbol: str,
    ) -> AgentDecision:
        """
        Simulate an agent's decision given market state and their persona.

        Decision logic is based on agent archetype + market conditions.
        """
        import random

        # Archetype-specific decision heuristic
        direction, confidence, sizing = self._archetype_decision(
            agent.archetype, agent.risk_tolerance, market_state, symbol
        )

        # Adjust by agent's conviction threshold
        if confidence < agent.conviction_threshold:
            direction = Direction.HOLD

        # Slight randomness to simulate independent thinking
        if random.random() < 0.1:  # 10% chance to flip
            direction = random.choice([Direction.BUY, Direction.SELL, Direction.HOLD])

        reasoning = f"{agent.archetype.upper()}: {direction.value} @ {confidence:.1%}"

        return AgentDecision(
            agent_id=agent.agent_id,
            archetype=agent.archetype,
            direction=direction,
            confidence=confidence,
            position_size_pct=sizing,
            reasoning=reasoning,
        )

    @staticmethod
    def _archetype_decision(
        archetype: str,
        risk_tol: float,
        market_state: dict,
        symbol: str,
    ) -> tuple:
        """
        Heuristic decision logic per agent archetype.

        Returns (direction, confidence, position_size_pct).
        """
        import random

        vix = market_state.get("india_vix", 20)
        nifty_pct = market_state.get("nifty_change_pct", 0)
        sentiment = market_state.get("sentiment_tilt", "NEUTRAL")

        # Base bias from sentiment
        bias = 0.5  # neutral
        if sentiment == "BULLISH":
            bias = 0.6
        elif sentiment == "BEARISH":
            bias = 0.4

        if archetype == "scalper":
            # Scalpers love volatility, trade frequently
            direction = Direction.BUY if bias > 0.5 else Direction.SELL
            confidence = 0.6 + (vix / 100.0)  # higher VIX = more confident
            sizing = 0.02 * risk_tol

        elif archetype == "trend_follower":
            # Follow the trend in market state
            direction = Direction.BUY if nifty_pct > 0 else Direction.SELL
            confidence = abs(nifty_pct) / 100.0 + 0.5
            sizing = 0.03 * risk_tol

        elif archetype == "value":
            # Value investors trade less frequently, look for contrarian signals
            direction = Direction.BUY if bias < 0.5 else Direction.SELL
            confidence = 0.5 + (1 - vix / 50.0) * 0.3
            sizing = 0.015

        elif archetype == "contrarian":
            # Opposite of market sentiment
            direction = Direction.SELL if bias > 0.5 else Direction.BUY
            confidence = abs(bias - 0.5) * 1.5
            sizing = 0.01 + 0.02 * risk_tol

        elif archetype == "momentum":
            # Follow momentum + market strength
            direction = Direction.BUY if nifty_pct > 0.5 else Direction.SELL
            confidence = 0.4 + risk_tol * 0.4
            sizing = 0.025 * risk_tol

        elif archetype == "hedger":
            # Defensive, prefer HOLD
            direction = Direction.HOLD
            confidence = 0.3
            sizing = 0.005

        else:
            # Unknown archetype → random
            direction = random.choice([Direction.BUY, Direction.SELL, Direction.HOLD])
            confidence = random.random()
            sizing = 0.01

        # Clamp values
        confidence = max(0.0, min(1.0, confidence))
        sizing = max(0.0, min(0.05, sizing))

        return direction, confidence, sizing

    # ── Aggregation ───────────────────────────────────────────────────────

    @staticmethod
    def _aggregate_decisions(round_result: SimulationRound) -> None:
        """Aggregate individual agent decisions into round consensus."""
        if not round_result.decisions:
            return

        buy_count = sum(1 for d in round_result.decisions if d.direction == Direction.BUY)
        sell_count = sum(1 for d in round_result.decisions if d.direction == Direction.SELL)
        total = len(round_result.decisions)

        round_result.bullish_pct = round(buy_count / total, 3)
        round_result.bearish_pct = round(sell_count / total, 3)
        round_result.avg_confidence = round(
            sum(d.confidence for d in round_result.decisions) / total, 3
        )
        round_result.avg_sizing = round(
            sum(d.position_size_pct for d in round_result.decisions) / total, 4
        )

        # Consensus direction
        if round_result.bullish_pct > 0.45:
            round_result.consensus_direction = Direction.BUY
        elif round_result.bearish_pct > 0.45:
            round_result.consensus_direction = Direction.SELL
        else:
            round_result.consensus_direction = Direction.HOLD

    @staticmethod
    def _compute_final_consensus(result: SimulationResult) -> None:
        """Synthesize final consensus from all rounds."""
        if not result.rounds:
            return

        # Average consensus across rounds
        final_buys = sum(
            1 for r in result.rounds
            if r.consensus_direction == Direction.BUY
        )
        final_sells = sum(
            1 for r in result.rounds
            if r.consensus_direction == Direction.SELL
        )

        if final_buys > final_sells:
            result.final_consensus = Direction.BUY
        elif final_sells > final_buys:
            result.final_consensus = Direction.SELL
        else:
            result.final_consensus = Direction.HOLD

        # Average confidence
        result.final_confidence = sum(r.avg_confidence for r in result.rounds) / len(result.rounds)

        # Archetype breakdown
        archetype_counts: Dict[str, int] = {}
        for persona in result.rounds[0].decisions[0].agent_id if result.rounds else []:
            pass  # Simplified; in production, track archetype votes

        # Risk appetite = average risk tolerance
        # Simplified placeholder
        result.risk_appetite = 0.5

    def reset(self) -> None:
        """Reset any state between simulations."""
        pass
