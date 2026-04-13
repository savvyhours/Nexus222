"""
NEXUS-II — PositionManager (v1.0 preserved + dynamic SL/TP)

Tracks all open positions and evaluates exit conditions on every price update.
Six exit triggers (unchanged from v1.0):
  1. Stop loss hit           — price crosses ATR-based SL
  2. Target achieved         — price reaches R×R target
  3. Trailing stop triggered — price retreats trailing_stop_atr × ATR from high
  4. Time-based exit         — intraday squareoff at 15:10 IST
  5. Agent signal reversal   — agent that opened the position now signals opposite
  6. Risk limit breach       — drawdown or daily loss limit hit

SL / TP multipliers come from WeightCalibrationAgent so they adapt automatically
to market volatility regime (e.g. wider SL in HIGH_VOL, tighter in LOW_VOL).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Optional
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    from core.calibration.weight_calibration_agent import WeightCalibrationAgent

log = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")

_INTRADAY_SQUAREOFF = datetime.now(IST).replace(hour=15, minute=10, second=0, microsecond=0)


# ── Data models ────────────────────────────────────────────────────────────────

@dataclass
class Position:
    """
    Represents a single open position.

    Fields are updated on every mark-to-market (price tick).
    """
    symbol:            str
    direction:         str          # "BUY" | "SELL"
    quantity:          int
    entry_price:       float
    current_price:     float
    stop_loss:         float        # absolute price
    target:            float        # absolute price
    trailing_stop:     float        # absolute trailing-stop price (moves with price)
    atr:               float        # ATR at entry — used to recalculate trailing stop
    product_type:      str          # "INTRADAY" | "CNC" | "OPTIONS"
    agent_name:        str
    strategy:          str
    opened_at:         datetime = field(default_factory=lambda: datetime.now(IST))
    high_since_entry:  float = 0.0  # tracks highest price since long entry
    low_since_entry:   float = 0.0  # tracks lowest price since short entry
    sector:            str  = ""
    order_value:       float = 0.0  # entry_price × quantity (for sector check)
    metadata:          dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.high_since_entry = self.entry_price
        self.low_since_entry  = self.entry_price
        if self.order_value == 0.0:
            self.order_value = self.entry_price * self.quantity

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized P&L in ₹."""
        if self.direction == "BUY":
            return (self.current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - self.current_price) * self.quantity

    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized P&L as fraction of cost basis."""
        cost = self.entry_price * self.quantity
        return self.unrealized_pnl / cost if cost > 0 else 0.0


@dataclass
class ExitDecision:
    """Result of should_exit() evaluation."""
    should_exit: bool
    reason:      str = ""
    exit_type:   str = ""  # "STOP_LOSS" | "TARGET" | "TRAILING_STOP" | "TIME" | "SIGNAL_REVERSAL" | "RISK_BREACH"
    urgency:     str = "NORMAL"  # "NORMAL" | "IMMEDIATE" (market order required)


# ── PositionManager ────────────────────────────────────────────────────────────

class PositionManager:
    """
    Tracks open positions and evaluates exit conditions (v1.0 preserved + dynamic SL/TP).

    Parameters
    ----------
    calibration_agent : WeightCalibrationAgent — source of dynamic SL/TP multipliers
    """

    def __init__(self, calibration_agent: "WeightCalibrationAgent") -> None:
        self._cal = calibration_agent
        self._positions: dict[str, Position] = {}   # symbol → Position
        # Portfolio delta for options positions (keep within ±0.3)
        self._portfolio_delta: float = 0.0

    # ── Position lifecycle ────────────────────────────────────────────────────

    def update_position(self, trade: dict) -> Position:
        """
        Open a new position from a filled trade dict.

        trade must contain: symbol, direction, quantity, entry_price, atr,
        stop_loss, target, trailing_stop, product_type, agent_name, strategy,
        sector (optional), metadata (optional), options_delta (optional).
        """
        symbol = trade["symbol"]
        pos = Position(
            symbol       = symbol,
            direction    = trade["direction"],
            quantity     = int(trade["quantity"]),
            entry_price  = float(trade["entry_price"]),
            current_price= float(trade["entry_price"]),
            stop_loss    = float(trade["stop_loss"]),
            target       = float(trade["target"]),
            trailing_stop= float(trade.get("trailing_stop", trade["stop_loss"])),
            atr          = float(trade.get("atr", 0.0)),
            product_type = trade.get("product_type", "INTRADAY"),
            agent_name   = trade.get("agent_name", ""),
            strategy     = trade.get("strategy", ""),
            sector       = trade.get("sector", ""),
            order_value  = float(trade["entry_price"]) * int(trade["quantity"]),
            metadata     = trade.get("metadata", {}),
        )
        self._positions[symbol] = pos

        # Track portfolio delta for options
        delta = float(trade.get("options_delta", 0.0))
        self._portfolio_delta += delta * pos.quantity

        log.info(
            "Position opened: %s %s × %d @ ₹%.2f | SL=₹%.2f | Target=₹%.2f",
            pos.direction, symbol, pos.quantity, pos.entry_price,
            pos.stop_loss, pos.target,
        )
        return pos

    def close_position(self, symbol: str, exit_price: float) -> Optional[dict]:
        """
        Remove position and return a trade-result dict for performance tracking.

        Returns None if symbol is not in open positions.
        """
        pos = self._positions.pop(symbol, None)
        if not pos:
            log.warning("PositionManager: close_position('%s') — not found", symbol)
            return None

        if pos.direction == "BUY":
            pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            pnl = (pos.entry_price - exit_price) * pos.quantity

        result = {
            "symbol":       symbol,
            "direction":    pos.direction,
            "quantity":     pos.quantity,
            "entry_price":  pos.entry_price,
            "exit_price":   exit_price,
            "pnl":          round(pnl, 2),
            "pnl_pct":      round(pnl / (pos.entry_price * pos.quantity), 4),
            "agent_name":   pos.agent_name,
            "strategy":     pos.strategy,
            "opened_at":    pos.opened_at,
            "closed_at":    datetime.now(IST),
            "hold_minutes": int((datetime.now(IST) - pos.opened_at).total_seconds() / 60),
            "metadata":     pos.metadata,
        }
        log.info(
            "Position closed: %s %s | PnL=₹%.2f (%.2f%%)",
            symbol, pos.direction, pnl, result["pnl_pct"] * 100,
        )
        return result

    def mark_to_market(self, symbol: str, current_price: float) -> None:
        """
        Update current price and trailing stop for an open position.

        Call on every price tick or periodic mark-to-market.
        """
        pos = self._positions.get(symbol)
        if not pos:
            return

        pos.current_price = current_price

        # Update high/low watermarks
        if current_price > pos.high_since_entry:
            pos.high_since_entry = current_price
        if current_price < pos.low_since_entry:
            pos.low_since_entry = current_price

        # Advance trailing stop (only moves in the favourable direction)
        if pos.atr > 0:
            # trailing_stop is set as absolute price at entry; keep cached multiplier
            trailing_atr = pos.metadata.get("trailing_stop_atr", 1.5)
            if pos.direction == "BUY":
                new_ts = pos.high_since_entry - trailing_atr * pos.atr
                if new_ts > pos.trailing_stop:
                    pos.trailing_stop = new_ts
            else:
                new_ts = pos.low_since_entry + trailing_atr * pos.atr
                if new_ts < pos.trailing_stop:
                    pos.trailing_stop = new_ts

    # ── Exit evaluation ───────────────────────────────────────────────────────

    def should_exit(
        self,
        symbol: str,
        current_price: float,
        agent_reversal: bool = False,
        risk_breach: bool = False,
    ) -> ExitDecision:
        """
        Evaluate all six exit conditions for `symbol`.

        Parameters
        ----------
        current_price  : latest market price
        agent_reversal : True if the originating agent now signals the opposite direction
        risk_breach    : True if DrawdownMonitor has tripped a circuit breaker

        Returns
        -------
        ExitDecision — check should_exit bool and exit_type for execution routing.
        """
        pos = self._positions.get(symbol)
        if not pos:
            return ExitDecision(should_exit=False)

        self.mark_to_market(symbol, current_price)

        # ── Trigger 6: Risk limit breach (highest priority) ──
        if risk_breach:
            return ExitDecision(
                should_exit=True, exit_type="RISK_BREACH", urgency="IMMEDIATE",
                reason="DrawdownMonitor circuit breaker triggered — immediate exit",
            )

        # ── Trigger 1: Stop loss ──
        if pos.direction == "BUY" and current_price <= pos.stop_loss:
            return ExitDecision(
                should_exit=True, exit_type="STOP_LOSS", urgency="IMMEDIATE",
                reason=f"Price ₹{current_price:.2f} ≤ SL ₹{pos.stop_loss:.2f}",
            )
        if pos.direction == "SELL" and current_price >= pos.stop_loss:
            return ExitDecision(
                should_exit=True, exit_type="STOP_LOSS", urgency="IMMEDIATE",
                reason=f"Price ₹{current_price:.2f} ≥ SL ₹{pos.stop_loss:.2f}",
            )

        # ── Trigger 2: Target achieved ──
        if pos.direction == "BUY" and current_price >= pos.target:
            return ExitDecision(
                should_exit=True, exit_type="TARGET", urgency="NORMAL",
                reason=f"Price ₹{current_price:.2f} ≥ Target ₹{pos.target:.2f}",
            )
        if pos.direction == "SELL" and current_price <= pos.target:
            return ExitDecision(
                should_exit=True, exit_type="TARGET", urgency="NORMAL",
                reason=f"Price ₹{current_price:.2f} ≤ Target ₹{pos.target:.2f}",
            )

        # ── Trigger 3: Trailing stop ──
        if pos.direction == "BUY" and current_price <= pos.trailing_stop:
            return ExitDecision(
                should_exit=True, exit_type="TRAILING_STOP", urgency="NORMAL",
                reason=f"Price ₹{current_price:.2f} ≤ trailing stop ₹{pos.trailing_stop:.2f}",
            )
        if pos.direction == "SELL" and current_price >= pos.trailing_stop:
            return ExitDecision(
                should_exit=True, exit_type="TRAILING_STOP", urgency="NORMAL",
                reason=f"Price ₹{current_price:.2f} ≥ trailing stop ₹{pos.trailing_stop:.2f}",
            )

        # ── Trigger 4: Time-based squareoff ──
        if pos.product_type == "INTRADAY":
            now = datetime.now(IST).time()
            if now >= _INTRADAY_SQUAREOFF.time():
                return ExitDecision(
                    should_exit=True, exit_type="TIME", urgency="IMMEDIATE",
                    reason=f"Intraday squareoff at 15:10 IST",
                )

        # ── Trigger 5: Agent signal reversal ──
        if agent_reversal:
            return ExitDecision(
                should_exit=True, exit_type="SIGNAL_REVERSAL", urgency="NORMAL",
                reason=f"Agent '{pos.agent_name}' has reversed its signal",
            )

        return ExitDecision(should_exit=False)

    # ── Dynamic SL / TP computation ───────────────────────────────────────────

    async def compute_stop_loss(
        self,
        entry_price: float,
        atr: float,
        direction: str = "BUY",
        use_positional: bool = False,
    ) -> float:
        """
        ATR-based stop loss using dynamic multiplier from WeightCalibrationAgent.

        Parameters
        ----------
        entry_price     : fill price
        atr             : ATR(14) at entry
        direction       : "BUY" or "SELL"
        use_positional  : use positional_sl_atr instead of intraday_sl_atr
        """
        multipliers = await self._cal.get_sl_tp_multipliers()
        key = "positional_sl_atr" if use_positional else "intraday_sl_atr"
        sl_atr = multipliers.get(key, 2.0)

        if direction == "BUY":
            return round(entry_price - atr * sl_atr, 2)
        else:
            return round(entry_price + atr * sl_atr, 2)

    async def compute_target(
        self,
        entry_price: float,
        stop_loss: float,
        direction: str = "BUY",
    ) -> float:
        """
        R-multiple target using dynamic risk:reward from WeightCalibrationAgent.
        """
        multipliers = await self._cal.get_sl_tp_multipliers()
        rr = multipliers.get("target_risk_reward", 2.0)
        risk = abs(entry_price - stop_loss)

        if direction == "BUY":
            return round(entry_price + risk * rr, 2)
        else:
            return round(entry_price - risk * rr, 2)

    async def compute_trailing_stop(
        self,
        entry_price: float,
        atr: float,
        direction: str = "BUY",
    ) -> float:
        """Initial trailing stop price at entry (advances with mark_to_market)."""
        multipliers = await self._cal.get_sl_tp_multipliers()
        ts_atr = multipliers.get("trailing_stop_atr", 1.5)

        if direction == "BUY":
            return round(entry_price - atr * ts_atr, 2)
        else:
            return round(entry_price + atr * ts_atr, 2)

    # ── Accessors ─────────────────────────────────────────────────────────────

    def get_open_positions(self) -> list[Position]:
        """All currently open positions."""
        return list(self._positions.values())

    def get_position_by_symbol(self, symbol: str) -> Optional[Position]:
        """Returns the open position for `symbol`, or None."""
        return self._positions.get(symbol)

    def get_portfolio_delta(self) -> float:
        """
        Net portfolio delta (for options positions).
        Target: maintain within ±0.3 (OptionsAgent responsibility).
        """
        return self._portfolio_delta

    def compute_unrealized_pnl(self) -> float:
        """Total unrealized P&L across all open positions (₹)."""
        return sum(p.unrealized_pnl for p in self._positions.values())

    def get_open_symbols(self) -> list[str]:
        """List of symbols with open positions."""
        return list(self._positions.keys())

    def position_count(self) -> int:
        """Number of open positions."""
        return len(self._positions)
