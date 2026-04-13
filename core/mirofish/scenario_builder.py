"""
NEXUS-II — MiroFish Scenario Builder
Constructs macroeconomic scenarios for weekend simulations.

The scenario builder takes:
  1. Current market state (Friday close, VIX, sector sentiment, FII flows)
  2. Macro seeds (user-injected hypothetical conditions: "RBI cuts rates", "crude +10%")
  3. Historical distributions of agent behavior

And produces a Scenario object that the simulation runner uses to initialize 100+ agents.

Usage:
    builder = ScenarioBuilder()
    scenario = await builder.build(
        market_state={...},
        macro_seeds=["RBI_RATE_CUT", "CRUDE_SPIKE"],
    )
    # Scenario with 100 agents initialized per their personas + market conditions
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")

# ── Macro seeds (user-selectable hypothetical conditions) ──────────────────────

class MacroSeed(str, Enum):
    """Hypothetical macro conditions to test agent consensus."""
    RBI_RATE_CUT = "RBI_RATE_CUT"
    RBI_RATE_HIKE = "RBI_RATE_HIKE"
    CRUDE_SPIKE = "CRUDE_SPIKE"
    CRUDE_CRASH = "CRUDE_CRASH"
    INR_WEAKNESS = "INR_WEAKNESS"
    INR_STRENGTH = "INR_STRENGTH"
    GEOPOLITICAL_CRISIS = "GEOPOLITICAL_CRISIS"
    TECH_EARNINGS_BEAT = "TECH_EARNINGS_BEAT"
    GROWTH_SLOWDOWN = "GROWTH_SLOWDOWN"
    MONSOON_FAILURE = "MONSOON_FAILURE"


# ── Agent persona ─────────────────────────────────────────────────────────────

@dataclass
class AgentPersona:
    """Pre-configured agent behavioral profile."""
    agent_id: str
    archetype: str  # "scalper", "trend_follower", "value", "contrarian", "momentum", "hedger"
    risk_tolerance: float  # 0.0 = conservative, 1.0 = aggressive
    memory_horizon: int  # days of historical memory
    sector_bias: str  # preferred sector or "none"
    conviction_threshold: float  # min confidence before betting

    @classmethod
    def generate_population(cls, count: int = 100) -> List[AgentPersona]:
        """Generate a diverse population of agent personas."""
        import random
        archetypes = ["scalper", "trend_follower", "value", "contrarian", "momentum", "hedger"]
        sectors = ["IT", "BANK", "AUTO", "PHARMA", "ENERGY", "none"]
        personas = []
        for i in range(count):
            personas.append(cls(
                agent_id=f"AGENT_{i:04d}",
                archetype=random.choice(archetypes),
                risk_tolerance=round(random.random(), 2),
                memory_horizon=random.randint(5, 60),
                sector_bias=random.choice(sectors),
                conviction_threshold=round(random.uniform(0.5, 0.9), 2),
            ))
        return personas


# ── Scenario ──────────────────────────────────────────────────────────────────

@dataclass
class Scenario:
    """Complete market scenario for weekend simulation."""
    scenario_id: str
    market_date: datetime
    market_state: dict  # Friday market snapshot
    macro_seeds: List[str]  # applied conditions
    agent_personas: List[AgentPersona]  # 100+ agents
    initial_positions: Dict[str, float] = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "market_date": self.market_date.isoformat(),
            "market_state": self.market_state,
            "macro_seeds": self.macro_seeds,
            "num_agents": len(self.agent_personas),
            "initial_positions": self.initial_positions,
            "description": self.description,
        }


# ── Scenario Builder ──────────────────────────────────────────────────────────

class ScenarioBuilder:
    """
    Constructs scenarios by combining market state + macro seeds + agent personas.

    Parameters
    ----------
    default_population_size : Number of agents to initialize (default 100).
    """

    def __init__(self, default_population_size: int = 100) -> None:
        self._pop_size = default_population_size

    async def build(
        self,
        market_state: dict,
        macro_seeds: Optional[List[str]] = None,
        agent_personas: Optional[List[AgentPersona]] = None,
        initial_positions: Optional[dict] = None,
    ) -> Scenario:
        """
        Build a scenario combining market conditions + macro seeds + agents.

        Parameters
        ----------
        market_state    : Market snapshot (VIX, NIFTY price, sector momentum, etc.)
        macro_seeds     : List of MacroSeed values (e.g. ["RBI_RATE_CUT"])
        agent_personas  : Custom agent population; generated if not provided.
        initial_positions : Seed positions (symbol → qty); default empty.

        Returns
        -------
        Scenario object ready for SimulationRunner.
        """
        import uuid

        seeds = macro_seeds or []
        self._validate_seeds(seeds)

        personas = (
            agent_personas or
            AgentPersona.generate_population(self._pop_size)
        )

        positions = initial_positions or {}

        scenario = Scenario(
            scenario_id=f"SIM-{uuid.uuid4().hex[:8].upper()}",
            market_date=datetime.now(IST),
            market_state=market_state,
            macro_seeds=seeds,
            agent_personas=personas,
            initial_positions=positions,
            description=self._describe(market_state, seeds),
        )

        log.info(
            "Built scenario %s: %d agents, seeds=%s",
            scenario.scenario_id, len(personas), seeds,
        )
        return scenario

    # ── Scenario modifiers ────────────────────────────────────────────────

    def apply_macro_seed_to_state(self, state: dict, seed: str) -> dict:
        """Apply a macro seed to the market state."""
        state = dict(state)  # shallow copy
        seed = seed.upper()

        if seed == "RBI_RATE_CUT":
            state["policy_rate_delta"] = -50
            state["sentiment_tilt"] = "BULLISH"
        elif seed == "RBI_RATE_HIKE":
            state["policy_rate_delta"] = +50
            state["sentiment_tilt"] = "BEARISH"
        elif seed == "CRUDE_SPIKE":
            state["crude_pct_change"] = 10.0
            state["inflation_risk"] = "HIGH"
        elif seed == "CRUDE_CRASH":
            state["crude_pct_change"] = -10.0
            state["inflation_risk"] = "LOW"
        elif seed == "INR_WEAKNESS":
            state["usd_inr_delta"] = 3.0
            state["export_tilt"] = "POSITIVE"
        elif seed == "INR_STRENGTH":
            state["usd_inr_delta"] = -3.0
            state["import_tilt"] = "POSITIVE"
        elif seed == "GEOPOLITICAL_CRISIS":
            state["tail_risk"] = "HIGH"
            state["vix_shock"] = 15
        elif seed == "TECH_EARNINGS_BEAT":
            state["sector_beats"] = {"IT": 3.0}
            state["sentiment_tilt"] = "BULLISH"
        elif seed == "GROWTH_SLOWDOWN":
            state["gdp_growth_signal"] = -1.5
            state["sentiment_tilt"] = "BEARISH"
        elif seed == "MONSOON_FAILURE":
            state["rural_sentiment"] = "BEARISH"
            state["commodity_exposure"] = "HIGHER"

        return state

    @staticmethod
    def _validate_seeds(seeds: List[str]) -> None:
        """Validate that all seeds are recognized MacroSeed values."""
        valid = {s.value for s in MacroSeed}
        for seed in seeds:
            if seed.upper() not in valid:
                log.warning("Unrecognized macro seed: %s", seed)

    @staticmethod
    def _describe(state: dict, seeds: List[str]) -> str:
        """Generate a human-readable scenario description."""
        vix = state.get("india_vix", "unknown")
        nifty_pct = state.get("nifty_change_pct", 0)
        seeds_str = " + ".join(seeds) if seeds else "baseline"
        return f"VIX={vix}, NIFTY {nifty_pct:+.1f}%, seeds: {seeds_str}"
