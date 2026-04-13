"""
NEXUS-II — core/agents package
Exports all agent classes and key data types for use by main.py and tests.
"""
from core.agents.base_agent import Action, AgentSignal, BaseAgent, PerformanceMetrics
from core.agents.scalper_agent import ScalperAgent
from core.agents.trend_agent import TrendFollowerAgent
from core.agents.mean_reversion_agent import MeanReversionAgent
from core.agents.sentiment_agent import SentimentAgent
from core.agents.fundamentals_agent import FundamentalsAgent
from core.agents.macro_agent import MacroAgent
from core.agents.options_agent import OptionsAgent
from core.agents.pattern_agent import PatternAgent
from core.agents.quant_agent import QuantAgent
from core.agents.etf_agent import ETFAgent
from core.agents.master_orchestrator import ConsensusSignal, MasterOrchestrator
from core.agents.portfolio_manager import PMDecision, PortfolioManager, TradeProposal

__all__ = [
    # Base
    "Action", "AgentSignal", "BaseAgent", "PerformanceMetrics",
    # Strategy agents
    "ScalperAgent", "TrendFollowerAgent", "MeanReversionAgent",
    "SentimentAgent", "FundamentalsAgent", "MacroAgent",
    "OptionsAgent", "PatternAgent", "QuantAgent", "ETFAgent",
    # Orchestration
    "ConsensusSignal", "MasterOrchestrator",
    # Portfolio management
    "PMDecision", "PortfolioManager", "TradeProposal",
]
