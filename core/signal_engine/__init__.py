"""
NEXUS-II v2.1 — signal_engine package
Layer 6: Dynamic signal scoring, 22 backtested strategies, candlestick patterns.

Public surface
--------------
DynamicSignalScorer   — regime-adaptive weighted scorer (7 components)
SignalResult          — final scored signal for one symbol
ComponentBreakdown    — per-component weight × raw contribution
build_components      — convenience factory for the components dict
debate_conviction_to_signed — convert DebateVerdict to signed score

StrategySignal        — single strategy output
run_all               — run all 22 strategies
active_signals        — filter to non-zero strategies
net_score             — [-1.0, +1.0] aggregate strategy score

PatternResult         — single pattern detection result
scan_all              — scan all candlestick patterns on a bar sequence
composite_score       — [-1.0, +1.0] summary score for scorer input
"""

from core.signal_engine.dynamic_signal_scorer import (
    DynamicSignalScorer,
    SignalResult,
    ComponentBreakdown,
    build_components,
    debate_conviction_to_signed,
    COMPONENT_KEYS,
)

from core.signal_engine.strategy_library import (
    StrategySignal,
    run_all,
    active_signals,
    net_score,
    STRATEGY_REGISTRY,
)

from core.signal_engine.candlestick_patterns import (
    PatternResult,
    scan_all,
    composite_score,
)

__all__ = [
    # scorer
    "DynamicSignalScorer",
    "SignalResult",
    "ComponentBreakdown",
    "build_components",
    "debate_conviction_to_signed",
    "COMPONENT_KEYS",
    # strategy library
    "StrategySignal",
    "run_all",
    "active_signals",
    "net_score",
    "STRATEGY_REGISTRY",
    # candlestick
    "PatternResult",
    "scan_all",
    "composite_score",
]
