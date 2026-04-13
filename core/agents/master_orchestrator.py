"""
NEXUS-II — MasterOrchestrator
Layer 2 coordinator: runs all 10 strategy sub-agents concurrently,
collects their AgentSignals, performs Sharpe-weighted voting per symbol,
and calls Claude Sonnet for LLM adjudication when the weighted vote is
close to the threshold.

Flow:
    run_cycle(universe, market_data_map, sentiment_data_map)
        → gather all agent.analyze() calls concurrently
        → per-symbol: compute weighted_vote using dynamic agent_weights
        → if |vote| >= signal_threshold → pass to Layer 3 (debate pipeline)
        → if vote ambiguous → LLM adjudication
        → return list[ConsensusSignal]
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

import anthropic

from config.settings import LLM_MAIN
from core.agents.base_agent import Action, AgentSignal, BaseAgent

if TYPE_CHECKING:
    from core.calibration.weight_calibration_agent import WeightCalibrationAgent

log = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")

_ADJUDICATION_ZONE = 0.10   # if vote is within threshold ± this → use LLM
_MAX_SIGNALS_PER_AGENT = 3  # cap signals per agent per symbol

_ADJUDICATION_SYSTEM = """\
You are the MasterOrchestrator adjudicator for NEXUS-II, an Indian market trading system.

Given a weighted vote score (positive=bullish, negative=bearish) that is near the
signal threshold, your job is to decide: BUY, SELL, or HOLD.

Consider:
- The strength and diversity of signals (do multiple independent agents agree?)
- Market regime context (trending vs mean-reverting)
- Risk asymmetry (in high-vol regimes, bias toward HOLD unless very clear)
- The current drawdown and daily P&L context

Return ONLY valid JSON:
{
  "decision": "BUY|SELL|HOLD",
  "conviction": 0.0-1.0,
  "reasoning": "1-2 sentence explanation"
}
"""


@dataclass
class ConsensusSignal:
    """Output from MasterOrchestrator — one per symbol that clears the threshold."""
    symbol:         str
    action:         Action
    weighted_vote:  float
    conviction:     float              # 0.0–1.0
    agent_signals:  list[AgentSignal]
    agent_weights:  dict[str, float]
    regime:         str
    timestamp:      datetime = field(default_factory=lambda: datetime.now(IST))
    reasoning:      str = ""
    adjudicated:    bool = False       # True if LLM tiebreaker was used


class MasterOrchestrator:
    """
    Coordinates all 10 strategy sub-agents and aggregates their votes
    into ConsensusSignals using dynamic Sharpe-weighted voting.
    """

    def __init__(
        self,
        agents: list[BaseAgent],
        calibration_agent: "WeightCalibrationAgent",
        claude_client: anthropic.AsyncAnthropic,
    ) -> None:
        self._agents = {a.AGENT_KEY: a for a in agents}
        self._calibration = calibration_agent
        self._claude = claude_client

    # ── Main entry point ──────────────────────────────────────────────────────

    async def run_cycle(
        self,
        universe: list[str],
        market_data_map: dict[str, dict],
        sentiment_data_map: dict[str, dict],
    ) -> list[ConsensusSignal]:
        """
        Full orchestration cycle for all symbols in universe.

        Returns list of ConsensusSignals that exceeded the signal threshold.
        """
        # Kill switch check
        if await self._calibration.is_kill_switch_active():
            log.warning("MasterOrchestrator: kill switch active — no new signals")
            return []

        # Get dynamic weights and threshold once (cached by calibration agent)
        agent_weights = await self._calibration.get_agent_weights()
        threshold     = await self._calibration.get_signal_threshold()
        regime        = (await self._calibration.get_current_regime()).value

        log.info(
            "Orchestrator cycle: %d symbols, threshold=%.2f, regime=%s",
            len(universe), threshold, regime,
        )

        # Run all agents on all symbols concurrently
        all_signals: dict[str, list[AgentSignal]] = {s: [] for s in universe}
        tasks = [
            self._run_agent_on_universe(agent, universe, market_data_map, sentiment_data_map)
            for agent in self._agents.values()
        ]
        agent_results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in agent_results:
            if isinstance(result, Exception):
                log.error("Agent task failed: %s", result)
                continue
            for symbol, signals in result.items():
                all_signals[symbol].extend(signals[:_MAX_SIGNALS_PER_AGENT])

        # Per-symbol: compute weighted vote and generate consensus
        consensus_signals: list[ConsensusSignal] = []
        for symbol in universe:
            sig_list = all_signals[symbol]
            if not sig_list:
                continue

            consensus = await self._compute_consensus(
                symbol, sig_list, agent_weights, threshold, regime,
            )
            if consensus is not None:
                consensus_signals.append(consensus)

        log.info("Orchestrator: %d consensus signal(s) generated", len(consensus_signals))
        return consensus_signals

    # ── Agent execution ───────────────────────────────────────────────────────

    async def _run_agent_on_universe(
        self,
        agent: BaseAgent,
        universe: list[str],
        market_data_map: dict[str, dict],
        sentiment_data_map: dict[str, dict],
    ) -> dict[str, list[AgentSignal]]:
        """Run one agent across all symbols; catch per-symbol errors."""
        results: dict[str, list[AgentSignal]] = {}
        for symbol in universe:
            try:
                md = market_data_map.get(symbol, {"symbol": symbol})
                sd = sentiment_data_map.get(symbol, {})
                signals = await agent.analyze(md, sd)
                results[symbol] = signals
            except Exception as exc:
                log.warning("Agent '%s' failed on %s: %s", agent.AGENT_KEY, symbol, exc)
                results[symbol] = []
        return results

    # ── Weighted voting ───────────────────────────────────────────────────────

    async def _compute_consensus(
        self,
        symbol: str,
        signals: list[AgentSignal],
        agent_weights: dict[str, float],
        threshold: float,
        regime: str,
    ) -> ConsensusSignal | None:
        """
        Compute Sharpe-weighted vote for a symbol.
        Returns ConsensusSignal if |vote| >= threshold, else None.
        Uses LLM adjudication if vote falls in ±_ADJUDICATION_ZONE of threshold.
        """
        # Weighted vote: Σ signal.strength × signal.action_numeric × agent_weight
        weighted_vote = 0.0
        total_weight  = 0.0
        for sig in signals:
            w = agent_weights.get(sig.agent_name, agent_weights.get(sig.strategy, 0.0))
            weighted_vote += sig.weighted_vote * w
            total_weight  += w

        if total_weight > 0:
            weighted_vote /= total_weight   # normalise to [-1, +1]

        abs_vote = abs(weighted_vote)
        log.debug("Symbol %s: weighted_vote=%.3f, threshold=%.2f", symbol, weighted_vote, threshold)

        # Clear signal: above threshold
        if abs_vote >= threshold:
            action = Action.BUY if weighted_vote > 0 else Action.SELL
            return ConsensusSignal(
                symbol=symbol, action=action,
                weighted_vote=weighted_vote,
                conviction=min(1.0, abs_vote),
                agent_signals=signals,
                agent_weights=agent_weights,
                regime=regime,
            )

        # Ambiguous zone: try LLM adjudication
        if abs_vote >= threshold - _ADJUDICATION_ZONE:
            return await self._llm_adjudicate(
                symbol, weighted_vote, signals, agent_weights, threshold, regime,
            )

        return None  # below threshold — no signal

    # ── LLM adjudication ──────────────────────────────────────────────────────

    async def _llm_adjudicate(
        self,
        symbol: str,
        weighted_vote: float,
        signals: list[AgentSignal],
        agent_weights: dict[str, float],
        threshold: float,
        regime: str,
    ) -> ConsensusSignal | None:
        """Claude Sonnet tiebreaker for near-threshold votes."""
        signal_summary = [
            {
                "agent": s.agent_name,
                "action": s.action.value,
                "strength": s.strength,
                "strategy": s.strategy,
                "reason": s.reason,
                "weight": agent_weights.get(s.agent_name, 0),
            }
            for s in signals
        ]

        prompt = (
            f"Symbol: {symbol}\n"
            f"Weighted vote: {weighted_vote:.3f} (threshold: {threshold:.2f})\n"
            f"Market regime: {regime}\n"
            f"Agent signals:\n{json.dumps(signal_summary, indent=2)}\n\n"
            "Decide: BUY, SELL, or HOLD."
        )

        try:
            response = await self._claude.messages.create(
                model=LLM_MAIN, max_tokens=256,
                system=_ADJUDICATION_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            if raw.startswith("```"):
                raw = "\n".join(l for l in raw.splitlines() if not l.startswith("```")).strip()
            data = json.loads(raw)

            decision  = data.get("decision", "HOLD").upper()
            conviction = float(data.get("conviction", 0.5))
            reasoning  = data.get("reasoning", "")

            if decision == "HOLD":
                return None

            action = Action.BUY if decision == "BUY" else Action.SELL
            log.info("LLM adjudicated %s → %s (conviction=%.2f): %s", symbol, decision, conviction, reasoning)
            return ConsensusSignal(
                symbol=symbol, action=action,
                weighted_vote=weighted_vote, conviction=conviction,
                agent_signals=signals, agent_weights=agent_weights,
                regime=regime, reasoning=reasoning, adjudicated=True,
            )

        except Exception as exc:
            log.error("LLM adjudication failed for %s: %s", symbol, exc)
            return None
