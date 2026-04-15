"""
NEXUS-II — BaseAgent
Abstract base class for all 10 strategy sub-agents.

Every strategy agent inherits from this class and implements `analyze()`.
Agent weights are NOT hardcoded — they are fetched dynamically from the
WeightCalibrationAgent before every consensus computation.

AgentSignal dataclass is the canonical inter-agent message format.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    from core.calibration.weight_calibration_agent import WeightCalibrationAgent

log = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")


# ── Action enum ───────────────────────────────────────────────────────────────

class Action(str, Enum):
    BUY  = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

    @property
    def numeric(self) -> float:
        """Numeric representation for weighted voting: BUY=+1, SELL=-1, HOLD=0."""
        return {"BUY": 1.0, "SELL": -1.0, "HOLD": 0.0}[self.value]


# ── AgentSignal dataclass ─────────────────────────────────────────────────────

@dataclass
class AgentSignal:
    """
    Canonical signal emitted by every strategy sub-agent.

    Fields
    ------
    symbol          : NSE/BSE ticker (e.g. "RELIANCE", "NIFTY25APR23000CE")
    action          : BUY | SELL | HOLD
    strength        : conviction 0.0–1.0 (used for weighted voting)
    entry           : suggested entry price (0.0 = use market price)
    stop_loss       : suggested stop-loss price
    target          : suggested target price
    position_size_pct : fraction of total capital to risk (0.01–0.10)
    reason          : 1-3 sentence human-readable rationale
    agent_name      : which sub-agent produced this signal
    strategy        : strategy name within the agent (e.g. "vwap_crossover")
    timestamp       : IST timestamp of signal generation
    metadata        : optional extra data (e.g. greeks for options signals)
    """
    symbol:            str   = ""
    action:            Action = Action.HOLD
    strength:          float  = 0.0    # 0.0–1.0
    entry:             float = 0.0
    stop_loss:         float = 0.0
    target:            float = 0.0
    position_size_pct: float = 0.03
    reason:            str  = ""
    agent_name:        str  = ""
    strategy:          str  = ""
    timestamp:         datetime = field(default_factory=lambda: datetime.now(IST))
    metadata:          dict = field(default_factory=dict)

    # ── Convenience ──────────────────────────────────────────────────────────

    @property
    def action_numeric(self) -> float:
        return self.action.numeric

    @property
    def weighted_vote(self) -> float:
        """strength × direction — used by MasterOrchestrator."""
        return self.strength * self.action_numeric

    def __post_init__(self) -> None:
        self.strength = max(0.0, min(1.0, self.strength))
        self.position_size_pct = max(0.01, min(0.10, self.position_size_pct))
        if not self.agent_name and self.strategy:
            self.agent_name = self.strategy


# ── Performance snapshot ──────────────────────────────────────────────────────

@dataclass
class PerformanceMetrics:
    """Rolling performance metrics updated after each trade closure."""
    sharpe_30d:   float = 0.0
    win_rate_30d: float = 0.0
    pnl_total:    float = 0.0
    trades_count: int   = 0
    wins:         int   = 0
    losses:       int   = 0

    def update(self, pnl: float) -> None:
        self.trades_count += 1
        self.pnl_total += pnl
        if pnl > 0:
            self.wins += 1
        else:
            self.losses += 1
        self.win_rate_30d = (
            self.wins / self.trades_count if self.trades_count else 0.0
        )


# ── BaseAgent ─────────────────────────────────────────────────────────────────

class BaseAgent(ABC):
    """
    Abstract base for all 10 NEXUS-II strategy sub-agents.

    Subclasses must implement:
        analyze(market_data, sentiment_data) → list[AgentSignal]

    Optionally override:
        get_entry_price(symbol, market_data)  → float
        get_stop_loss(symbol, market_data)    → float
        get_target(symbol, market_data)       → float

    The `weight` property is DYNAMIC — it delegates to the
    WeightCalibrationAgent and must NOT be hardcoded in subclasses.
    """

    #: Unique agent key used in calibration agent_weights dict.
    #: Must match keys in DEFAULT_AGENT_WEIGHTS (strategy_params.py).
    AGENT_KEY: str = ""

    def __init__(
        self,
        calibration_agent: "WeightCalibrationAgent",
        mcp_tools: Any = None,
        *,
        name: Optional[str] = None,
    ) -> None:
        self._calibration = calibration_agent
        self._tools = mcp_tools
        self.name: str = name or self.AGENT_KEY or self.__class__.__name__
        self.performance = PerformanceMetrics()
        self._trade_history: list[dict] = []
        log.info("Agent '%s' initialised", self.name)

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    async def analyze(
        self,
        market_data: dict,
        sentiment_data: dict,
    ) -> list[AgentSignal]:
        """
        Core strategy logic.

        Parameters
        ----------
        market_data    : dict with at minimum keys:
                         symbol, ohlcv (list of OHLCV bars), indicators
                         (RSI, MACD, ATR, ADX, EMA, BB, VWAP, OBV),
                         quote (current bid/ask/ltp), depth
        sentiment_data : dict with keys: score (-1..1), mentions, velocity,
                         headlines (list of str)

        Returns
        -------
        list[AgentSignal] — empty list = no signal this cycle
        """

    # ── Concrete helpers (may be overridden) ──────────────────────────────────

    async def get_entry_price(self, symbol: str, market_data: dict) -> float:
        """Default: current LTP. Override for limit-order strategies."""
        quote = market_data.get("quote", {})
        return float(quote.get("ltp", 0.0))

    async def get_stop_loss(self, symbol: str, market_data: dict) -> float:
        """
        Default ATR-based SL. Multiplier comes from WeightCalibrationAgent
        so it adapts to market volatility regime.
        """
        multipliers = await self._calibration.get_sl_tp_multipliers()
        atr = float(market_data.get("indicators", {}).get("atr", 0.0))
        ltp = await self.get_entry_price(symbol, market_data)
        return ltp - atr * multipliers.get("intraday_sl_atr", 2.0)

    async def get_target(self, symbol: str, market_data: dict) -> float:
        """Default 2R target (risk × target_risk_reward)."""
        multipliers = await self._calibration.get_sl_tp_multipliers()
        rr = multipliers.get("target_risk_reward", 2.0)
        entry = await self.get_entry_price(symbol, market_data)
        sl    = await self.get_stop_loss(symbol, market_data)
        risk  = abs(entry - sl)
        return entry + risk * rr

    # ── Dynamic weight ────────────────────────────────────────────────────────

    async def get_weight(self) -> float:
        """
        Returns this agent's current vote weight from WeightCalibrationAgent.
        Weight = 0.0 if 30-day Sharpe < 0 (muted by calibration agent).
        """
        weights = await self._calibration.get_agent_weights()
        return weights.get(self.AGENT_KEY, 0.0)

    # ── Performance tracking ──────────────────────────────────────────────────

    def update_performance(self, trade_result: dict) -> None:
        """
        Called by the execution layer after a trade is closed.

        trade_result must contain: symbol, pnl, entry_price, exit_price,
        direction, strategy, opened_at, closed_at.
        """
        pnl = float(trade_result.get("pnl", 0.0))
        self.performance.update(pnl)
        self._trade_history.append(trade_result)
        log.debug(
            "Agent '%s' trade closed: pnl=%.2f win_rate=%.1f%%",
            self.name, pnl, self.performance.win_rate_30d * 100,
        )

    # ── Utility ───────────────────────────────────────────────────────────────

    def _make_signal(
        self,
        symbol: str,
        action: Action,
        strength: float,
        reason: str,
        strategy: str = "",
        entry: float = 0.0,
        stop_loss: float = 0.0,
        target: float = 0.0,
        position_size_pct: float = 0.03,
        metadata: Optional[dict] = None,
    ) -> AgentSignal:
        """Convenience factory so subclasses write less boilerplate."""
        return AgentSignal(
            symbol=symbol,
            action=action,
            strength=strength,
            entry=entry,
            stop_loss=stop_loss,
            target=target,
            position_size_pct=position_size_pct,
            reason=reason,
            agent_name=self.name,
            strategy=strategy or self.AGENT_KEY,
            metadata=metadata or {},
        )

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} name='{self.name}' "
            f"trades={self.performance.trades_count} "
            f"win_rate={self.performance.win_rate_30d:.1%}>"
        )
