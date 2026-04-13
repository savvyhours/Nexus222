"""
NEXUS-II — Tier-2 Research Agents
Bull, Bear, and Risk researchers synthesize Tier-1 analyst reports into
coherent investment theses before the Debate Arena (Tier 3).
"""
from core.agents.researchers.bull_researcher import BullResearcher
from core.agents.researchers.bear_researcher import BearResearcher
from core.agents.researchers.risk_researcher import RiskResearcher

__all__ = ["BullResearcher", "BearResearcher", "RiskResearcher"]
