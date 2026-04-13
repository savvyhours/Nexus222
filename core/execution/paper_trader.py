"""
NEXUS-II — Paper Trader
Simulated execution engine for PAPER_TRADE mode.

All orders are immediately "filled" at the requested price (or last known price
if no price is specified). No real capital is moved. Tracks simulated positions
and P&L for the session.

Design notes:
  • Fills are assumed instant at the limit/market price specified in the request.
  • Slippage is configurable (default 0.0 — zero slippage for paper purity).
  • Positions are tracked as a dict: symbol → net qty (positive = long).
  • P&L is mark-to-market against fill prices; realised on position close.
  • Auto-approve timeout (5 min) is handled upstream by the Telegram bot;
    PaperTrader itself just fills without human gating.

Usage:
    paper = PaperTrader(slippage_pct=0.01)
    receipt = await paper.simulate(order_request)
"""
from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from core.execution.order_manager import ExecutionReceipt, OrderRequest, OrderStatus

log = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")


# ── Simulated position ────────────────────────────────────────────────────────

@dataclass
class SimPosition:
    symbol: str
    qty: int                    # positive = long, negative = short
    avg_price: float            # average entry price
    realised_pnl: float = 0.0
    open_ts: datetime = field(default_factory=lambda: datetime.now(IST))

    def mark_to_market(self, current_price: float) -> float:
        """Unrealised P&L at the given price."""
        return (current_price - self.avg_price) * self.qty


# ── Paper Trader ──────────────────────────────────────────────────────────────

class PaperTrader:
    """
    Simulates order execution with in-memory position and P&L tracking.

    Parameters
    ----------
    slippage_pct : Simulated slippage as a fraction of price (e.g. 0.001 = 0.1%).
                   Applied against the direction of the trade (buys fill slightly
                   higher, sells slightly lower).
    default_fill_price : Fallback fill price when no price is in the request.
                         In production, the caller should pass the last market price.
    """

    def __init__(
        self,
        slippage_pct: float = 0.0,
        default_fill_price: float = 100.0,
    ) -> None:
        self._slippage = slippage_pct
        self._default_price = default_fill_price
        # symbol → SimPosition (net, not per-leg)
        self._positions: Dict[str, SimPosition] = {}
        # full trade log
        self._trades: List[dict] = []
        self._cumulative_pnl: float = 0.0

    # ── Core simulate method ──────────────────────────────────────────────

    async def simulate(self, req: OrderRequest) -> ExecutionReceipt:
        """
        Immediately fill an order in simulation.

        Parameters
        ----------
        req : OrderRequest from OrderManager.

        Returns
        -------
        ExecutionReceipt with status=SIMULATED and a synthetic order_id.
        """
        fill_price = self._compute_fill_price(req)
        order_id = f"PAPER-{uuid.uuid4().hex[:10].upper()}"

        self._update_position(req.symbol, req.transaction_type, req.quantity, fill_price)
        pnl_delta = self._realised_pnl_on_fill(req.symbol, req.transaction_type, req.quantity, fill_price)
        self._cumulative_pnl += pnl_delta

        trade_entry = {
            "order_id": order_id,
            "symbol": req.symbol,
            "transaction_type": req.transaction_type,
            "quantity": req.quantity,
            "requested_price": req.price,
            "fill_price": fill_price,
            "pnl_delta": pnl_delta,
            "agent_name": req.agent_name,
            "strategy": req.strategy,
            "ts": datetime.now(IST).isoformat(),
        }
        self._trades.append(trade_entry)

        log.info(
            "[PAPER] %s %s × %d @ ₹%.2f (slippage=%.2f%%) | session P&L: ₹%.2f",
            req.transaction_type, req.symbol, req.quantity,
            fill_price, self._slippage * 100, self._cumulative_pnl,
        )

        return ExecutionReceipt(
            order_id=order_id,
            client_ref=req.client_ref,
            symbol=req.symbol,
            transaction_type=req.transaction_type,
            quantity=req.quantity,
            price=req.price,
            fill_price=fill_price,
            fill_qty=req.quantity,
            status=OrderStatus.SIMULATED,
        )

    # ── Position tracking ─────────────────────────────────────────────────

    def _compute_fill_price(self, req: OrderRequest) -> float:
        """Apply slippage to the requested price."""
        base = req.price or self._default_price
        if req.transaction_type == "BUY":
            return round(base * (1 + self._slippage), 4)
        else:
            return round(base * (1 - self._slippage), 4)

    def _realised_pnl_on_fill(
        self,
        symbol: str,
        transaction_type: str,
        qty: int,
        fill_price: float,
    ) -> float:
        """
        Compute realised P&L when closing (partially or fully) an existing position.
        Returns 0 for new positions or additions.
        """
        pos = self._positions.get(symbol)
        if pos is None:
            return 0.0
        is_closing = (
            (transaction_type == "SELL" and pos.qty > 0) or
            (transaction_type == "BUY"  and pos.qty < 0)
        )
        if not is_closing:
            return 0.0
        closed_qty = min(abs(pos.qty), qty)
        direction = 1 if pos.qty > 0 else -1
        pnl = (fill_price - pos.avg_price) * closed_qty * direction
        return round(pnl, 2)

    def _update_position(
        self,
        symbol: str,
        transaction_type: str,
        qty: int,
        fill_price: float,
    ) -> None:
        """Update in-memory position after a fill (FIFO average cost)."""
        signed_qty = qty if transaction_type == "BUY" else -qty
        pos = self._positions.get(symbol)

        if pos is None:
            self._positions[symbol] = SimPosition(
                symbol=symbol, qty=signed_qty, avg_price=fill_price
            )
            return

        new_qty = pos.qty + signed_qty

        if new_qty == 0:
            # position fully closed
            del self._positions[symbol]
            return

        if (pos.qty > 0 and signed_qty > 0) or (pos.qty < 0 and signed_qty < 0):
            # adding to existing position — recalculate average
            total_cost = pos.avg_price * abs(pos.qty) + fill_price * abs(signed_qty)
            pos.avg_price = round(total_cost / abs(new_qty), 4)
        # else: partial close — avg_price stays the same

        pos.qty = new_qty
        if abs(new_qty) < abs(pos.qty + signed_qty):
            # position flipped — reset avg price to fill price
            pos.avg_price = fill_price

    # ── Read-only accessors ───────────────────────────────────────────────

    @property
    def positions(self) -> Dict[str, dict]:
        """Return current open positions as plain dicts."""
        return {
            sym: {
                "qty": p.qty,
                "avg_price": p.avg_price,
                "symbol": p.symbol,
            }
            for sym, p in self._positions.items()
        }

    @property
    def cumulative_pnl(self) -> float:
        """Cumulative realised P&L for the session (₹)."""
        return self._cumulative_pnl

    def get_trade_log(self) -> List[dict]:
        """Full list of simulated trades this session."""
        return list(self._trades)

    def get_position(self, symbol: str) -> Optional[SimPosition]:
        """Return the SimPosition for a symbol, or None if flat."""
        return self._positions.get(symbol.upper())

    def unrealised_pnl(self, prices: Dict[str, float]) -> float:
        """
        Compute total unrealised P&L given a dict of current prices.

        Parameters
        ----------
        prices : {symbol → current_price}
        """
        total = 0.0
        for sym, pos in self._positions.items():
            price = prices.get(sym)
            if price:
                total += pos.mark_to_market(price)
        return round(total, 2)

    def reset(self) -> None:
        """Clear all state (call between sessions or on test teardown)."""
        self._positions.clear()
        self._trades.clear()
        self._cumulative_pnl = 0.0
        log.info("[PAPER] PaperTrader state reset.")
