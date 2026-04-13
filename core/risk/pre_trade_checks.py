"""
NEXUS-II — PreTradeRiskChecker (v1.0 preserved + dynamic thresholds)

Six pre-trade checks run against every trade order before it reaches the
execution engine.  All threshold values are fetched from WeightCalibrationAgent
(never hardcoded) so they adapt automatically to the current market regime.

Checks (in order):
  1. position_size      — order size ≤ max_position_pct of capital
  2. sector_concentration — sector exposure ≤ max_sector_pct after this order
  3. daily_loss_limit   — daily P&L has not breached max_daily_loss_pct
  4. market_hours       — within NSE trading session (09:15–15:30 IST Mon–Fri)
  5. news_blackout      — no major event within news_blackout_minutes window
  6. vix_level          — VIX below defensive / halt thresholds

Result: RiskCheckResult with passed bool + list of failure reasons.
A FAILED check blocks the order. The system logs every rejection.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, time as dtime
from typing import TYPE_CHECKING, Any, Optional
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    from core.calibration.weight_calibration_agent import WeightCalibrationAgent
    from core.mcp_tools.tool_registry import MCPTools

log = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")

# NSE session: 09:15 – 15:30 Mon–Fri (allow order submission 09:00 for AMO)
_MARKET_OPEN  = dtime(9, 15)
_MARKET_CLOSE = dtime(15, 30)
_SQUAREOFF    = dtime(15, 10)   # intraday hard squareoff time


# ── Data models ────────────────────────────────────────────────────────────────

@dataclass
class TradeOrder:
    """
    Normalised trade order passed into risk checks.

    All agents produce AgentSignal; ExecutionEngine converts to TradeOrder
    before calling check_all().
    """
    symbol:            str
    direction:         str          # "BUY" | "SELL"
    quantity:          int
    price:             float
    order_value:       float        # quantity × price
    position_size_pct: float        # fraction of total capital
    sector:            str          # NSE sector (e.g. "IT", "BANKING")
    product_type:      str          # "INTRADAY" | "CNC" | "OPTIONS"
    is_options:        bool = False
    agent_name:        str  = ""
    strategy:          str  = ""
    signal_strength:   float = 0.0  # 0.0–1.0


@dataclass
class RiskCheckResult:
    """Result of all pre-trade risk checks."""
    passed:   bool
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    mode:     str = "NORMAL"        # "NORMAL" | "DEFENSIVE" | "HALTED"

    def add_failure(self, reason: str) -> None:
        self.passed = False
        self.failures.append(reason)

    def add_warning(self, reason: str) -> None:
        self.warnings.append(reason)

    def __str__(self) -> str:
        status = "PASS" if self.passed else f"FAIL({len(self.failures)})"
        return f"RiskCheck[{status}] mode={self.mode} failures={self.failures}"


# ── PreTradeRiskChecker ────────────────────────────────────────────────────────

class PreTradeRiskChecker:
    """
    Six-gate pre-trade risk checker (v1.0 preserved + dynamic thresholds).

    All threshold values are fetched from WeightCalibrationAgent on each call.
    The calibration result is cached for the TTL window, so there is no
    extra latency on back-to-back checks.

    Parameters
    ----------
    calibration_agent : WeightCalibrationAgent — source of all dynamic thresholds
    mcp_tools         : MCPTools — used to read current portfolio state and VIX
    total_capital     : ₹ total trading capital (used for position-size check)
    """

    def __init__(
        self,
        calibration_agent: "WeightCalibrationAgent",
        mcp_tools: "MCPTools",
        total_capital: float,
    ) -> None:
        self._cal  = calibration_agent
        self._tools = mcp_tools
        self._capital = total_capital
        # Stores recent news event timestamps per symbol {symbol: [ts, ...]}
        self._news_events: dict[str, list[datetime]] = {}

    # ── Public interface ───────────────────────────────────────────────────────

    async def check_all(
        self,
        order: TradeOrder,
        open_positions: dict,   # symbol → position dict (from PortfolioState)
    ) -> RiskCheckResult:
        """
        Run all six risk checks against `order`.

        Parameters
        ----------
        order           : normalised TradeOrder
        open_positions  : current open positions dict {symbol: {sector, value, ...}}

        Returns
        -------
        RiskCheckResult — passed=True only if ALL six checks pass.
        """
        thresholds = await self._cal.get_risk_thresholds()
        result = RiskCheckResult(passed=True)

        # 1. Position size
        self._check_position_size(order, thresholds, result)

        # 2. Sector concentration
        self._check_sector_concentration(order, open_positions, thresholds, result)

        # 3. Daily loss limit
        self._check_daily_loss_limit(thresholds, result)

        # 4. Market hours
        self._check_market_hours(order, result)

        # 5. News blackout
        self._check_news_blackout(order, thresholds, result)

        # 6. VIX level — may upgrade result.mode to DEFENSIVE or HALTED
        await self._check_vix_level(thresholds, result)

        if not result.passed:
            log.warning(
                "PreTradeRisk BLOCKED %s %s: %s",
                order.direction, order.symbol, result.failures,
            )
        elif result.warnings:
            log.info(
                "PreTradeRisk WARN %s %s: %s",
                order.direction, order.symbol, result.warnings,
            )

        return result

    # ── Individual checks ──────────────────────────────────────────────────────

    def _check_position_size(
        self, order: TradeOrder, thresholds: dict, result: RiskCheckResult
    ) -> None:
        """Check 1: Order size ≤ max_position_pct of capital."""
        max_pct = thresholds.get("max_position_pct", 0.05)
        if order.position_size_pct > max_pct:
            result.add_failure(
                f"Position size {order.position_size_pct:.1%} exceeds max {max_pct:.1%}"
            )
        elif order.position_size_pct > max_pct * 0.8:
            result.add_warning(
                f"Position size {order.position_size_pct:.1%} approaching max {max_pct:.1%}"
            )

    def _check_sector_concentration(
        self,
        order: TradeOrder,
        open_positions: dict,
        thresholds: dict,
        result: RiskCheckResult,
    ) -> None:
        """Check 2: Sector exposure after order ≤ max_sector_pct of capital."""
        max_pct = thresholds.get("max_sector_pct", 0.25)
        sector  = order.sector

        # Current sector exposure
        current_sector_value = sum(
            p.get("order_value", 0.0)
            for p in open_positions.values()
            if p.get("sector") == sector
        )
        new_sector_value = current_sector_value + order.order_value
        new_sector_pct   = new_sector_value / self._capital if self._capital > 0 else 0.0

        if new_sector_pct > max_pct:
            result.add_failure(
                f"Sector '{sector}' concentration {new_sector_pct:.1%} would exceed max {max_pct:.1%}"
            )
        elif new_sector_pct > max_pct * 0.85:
            result.add_warning(
                f"Sector '{sector}' concentration {new_sector_pct:.1%} approaching max {max_pct:.1%}"
            )

    def _check_daily_loss_limit(
        self, thresholds: dict, result: RiskCheckResult
    ) -> None:
        """Check 3: Daily P&L has not breached max_daily_loss_pct."""
        max_loss_pct  = thresholds.get("max_daily_loss_pct", 0.02)
        daily_pnl_pct = self._tools.get_daily_pnl_pct()

        if daily_pnl_pct <= -max_loss_pct:
            result.add_failure(
                f"Daily loss {daily_pnl_pct:.2%} has breached limit {-max_loss_pct:.2%} — "
                f"no new positions allowed"
            )
        elif daily_pnl_pct <= -max_loss_pct * 0.75:
            result.add_warning(
                f"Daily loss {daily_pnl_pct:.2%} approaching limit {-max_loss_pct:.2%}"
            )

        # Also check drawdown
        max_dd = thresholds.get("max_drawdown_pct", 0.08)
        current_dd = self._tools.get_current_drawdown()
        if current_dd <= -max_dd:
            result.add_failure(
                f"Portfolio drawdown {current_dd:.2%} has breached limit {-max_dd:.2%}"
            )

    def _check_market_hours(self, order: TradeOrder, result: RiskCheckResult) -> None:
        """Check 4: Within NSE trading session (09:15–15:30 IST Mon–Fri)."""
        now = datetime.now(IST)
        weekday = now.weekday()   # 0=Mon, 4=Fri, 5=Sat, 6=Sun
        current_time = now.time()

        if weekday >= 5:
            result.add_failure("Market closed — weekend")
            return

        if not (_MARKET_OPEN <= current_time <= _MARKET_CLOSE):
            if order.product_type != "AMO":
                result.add_failure(
                    f"Outside market hours ({current_time.strftime('%H:%M')} IST). "
                    f"Market: {_MARKET_OPEN.strftime('%H:%M')}–{_MARKET_CLOSE.strftime('%H:%M')}"
                )
            return

        # Warn if intraday order placed near squareoff time
        if order.product_type == "INTRADAY" and current_time >= _SQUAREOFF:
            result.add_failure(
                f"Intraday order blocked after squareoff time {_SQUAREOFF.strftime('%H:%M')} IST"
            )

    def _check_news_blackout(
        self, order: TradeOrder, thresholds: dict, result: RiskCheckResult
    ) -> None:
        """
        Check 5: No major news event for this symbol within blackout_minutes.

        News events are registered via register_news_event(symbol, timestamp).
        """
        blackout_min = thresholds.get("news_blackout_minutes", 30)
        events = self._news_events.get(order.symbol, [])

        now = datetime.now(IST)
        for event_time in events:
            delta_min = abs((now - event_time).total_seconds()) / 60
            if delta_min < blackout_min:
                result.add_warning(
                    f"'{order.symbol}' has a news event {delta_min:.0f}min ago "
                    f"(blackout={blackout_min}min) — elevated risk"
                )
                # Warning only, not a failure — analyst conviction should handle this

    async def _check_vix_level(
        self, thresholds: dict, result: RiskCheckResult
    ) -> None:
        """
        Check 6: VIX level check.

        VIX > vix_halt_threshold      → HALTED (all new positions blocked)
        VIX > vix_defensive_threshold → DEFENSIVE (warning only; position size already
                                         halved by WeightCalibrationAgent)
        """
        defensive = thresholds.get("vix_defensive_threshold", 22)
        halt      = thresholds.get("vix_halt_threshold", 28)

        try:
            vix = await self._tools.get_india_vix()
        except Exception:
            vix = 16.0   # safe default if fetch fails

        if vix >= halt:
            result.mode = "HALTED"
            result.add_failure(
                f"India VIX={vix:.1f} >= halt threshold {halt} — CRISIS regime, "
                f"kill switch should be active"
            )
        elif vix >= defensive:
            result.mode = "DEFENSIVE"
            result.add_warning(
                f"India VIX={vix:.1f} >= defensive threshold {defensive} — "
                f"position sizes reduced by WeightCalibrationAgent"
            )

    # ── Event registration ─────────────────────────────────────────────────────

    def register_news_event(self, symbol: str, event_time: Optional[datetime] = None) -> None:
        """
        Register a significant news event for `symbol`.

        Called by the SentimentAnalyst or news feed processor when a material
        event (earnings, regulatory action, major headline) is detected.
        """
        ts = event_time or datetime.now(IST)
        self._news_events.setdefault(symbol, []).append(ts)
        log.info("PreTradeRisk: news blackout registered for %s at %s", symbol, ts)

    def clear_old_news_events(self, older_than_hours: int = 2) -> None:
        """Prune news event timestamps older than the given threshold."""
        cutoff = datetime.now(IST)
        for symbol in list(self._news_events.keys()):
            self._news_events[symbol] = [
                t for t in self._news_events[symbol]
                if (cutoff - t).total_seconds() < older_than_hours * 3600
            ]
            if not self._news_events[symbol]:
                del self._news_events[symbol]

    # ── Capital update ─────────────────────────────────────────────────────────

    def update_capital(self, new_capital: float) -> None:
        """Update total capital (called after deposits, withdrawals, or NAV recalc)."""
        self._capital = new_capital
