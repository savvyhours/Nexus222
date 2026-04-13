"""
NEXUS-II — core.mirofish
Weekend scenario simulation: 100+ agent swarm consensus extraction.
Runs Friday evening to generate conviction insights for Monday trading.
"""
from core.mirofish.scenario_builder import (
    AgentPersona,
    MacroSeed,
    Scenario,
    ScenarioBuilder,
)
from core.mirofish.simulation_runner import (
    Direction,
    AgentDecision,
    SimulationRound,
    SimulationResult,
    SimulationRunner,
)
from core.mirofish.report_extractor import (
    ConvictionScore,
    MiroFishReport,
    ReportExtractor,
)

__all__ = [
    "ScenarioBuilder",
    "Scenario",
    "AgentPersona",
    "MacroSeed",
    "SimulationRunner",
    "SimulationResult",
    "SimulationRound",
    "AgentDecision",
    "Direction",
    "ReportExtractor",
    "MiroFishReport",
    "ConvictionScore",
]
