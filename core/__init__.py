"""
NEXUS-II v2.1 — core package
Top-level re-exports for the most commonly used cross-layer types.

Sub-packages
────────────
core.agents          — 10 strategy sub-agents + orchestration + TradingAgents 4-tier
core.brain           — Qlib ML + FinRL trainer + model registry
core.calibration     — WeightCalibrationAgent + RegimeDetector + SafetyBounds
core.execution       — DhanExecutor (LIVE) + PaperTrader + OrderManager
core.mcp_tools       — MCP tool registry + data/compute/dhan tool sets
core.mirofish        — Monte-Carlo scenario simulation (risk stress-test)
core.monitoring      — GuardWatchdog + TelegramBot + ReportGenerator
core.risk            — PreTradeChecks + PositionManager + DrawdownMonitor
core.sentiment       — FinBERT engine + LLM enricher + SectorAggregator
core.signal_engine   — DynamicSignalScorer + 22 strategies + candlestick patterns
"""

# Expose the two most-imported calibration types at `core.` level
from core.calibration.weight_calibration_agent import WeightCalibrationAgent
from core.calibration.regime_detector import RegimeDetector

__all__ = [
    "WeightCalibrationAgent",
    "RegimeDetector",
]
