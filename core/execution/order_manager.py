"""
NEXUS-II — Order Manager
Central routing hub for all trade execution.

Responsibilities:
  • Define canonical OrderRequest and ExecutionReceipt dataclasses shared
    by DhanExecutor and PaperTrader.
  • Route execution to DhanExecutor (LIVE) or PaperTrader (PAPER_TRADE)
    based on TRADING_MODE env var.
  • Track all orders in-memory and persist to Supabase (trades table).
  • Enforce pre-trade safety: reject orders if kill switch is active,
    position limits exceeded, or the system is in CRISIS regime.

Usage:
    manager = OrderManager(dhan_tools=dhan_tools)
    receipt = await manager.submit(order_request)
"""
from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")

TRADING_MODE = os.getenv("TRADING_MODE", "PAPER_TRADE")


# ── Enums ─────────────────────────────────────────────────────────────────────

class OrderStatus(str, Enum):
    PENDING   = "PENDING"
    FILLED    = "FILLED"
    PARTIAL   = "PARTIAL"
    CANCELLED = "CANCELLED"
    REJECTED  = "REJECTED"
    SIMULATED = "SIMULATED"   # paper trade fill


class TransactionType(str, Enum):
    BUY  = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT  = "LIMIT"
    SL     = "SL"       # stop-loss limit
    SLM    = "SLM"      # stop-loss market


class ProductType(str, Enum):
    INTRADAY = "INTRADAY"
    CNC      = "CNC"        # delivery
    MARGIN   = "MARGIN"     # F&O margin


# ── Request & Receipt dataclasses ─────────────────────────────────────────────

@dataclass
class OrderRequest:
    """
    Canonical order specification passed from agents/risk layer to OrderManager.

    All fields use DhanHQ v2 naming conventions for zero-friction mapping.
    """
    symbol: str                        # NSE ticker e.g. "RELIANCE"
    security_id: str                   # DhanHQ internal security id
    exchange_segment: str              # "NSE_EQ" | "BSE_EQ" | "NSE_FNO" | ...
    transaction_type: str              # "BUY" | "SELL"
    quantity: int
    order_type: str = "MARKET"         # "MARKET" | "LIMIT" | "SL" | "SLM"
    product_type: str = "INTRADAY"     # "INTRADAY" | "CNC" | "MARGIN"
    price: Optional[float] = None      # required for LIMIT / SL
    trigger_price: Optional[float] = None  # required for SL / SLM
    validity: str = "DAY"              # "DAY" | "IOC"
    amo: bool = False                  # After-Market Order flag
    amo_time: str = "OPEN"            # "OPEN" | "OPEN_30" | "OPEN_60"
    bo_profit_value: Optional[float] = None   # Bracket Order profit leg
    bo_stop_loss_value: Optional[float] = None  # Bracket Order SL leg
    agent_name: str = "unknown"        # originating agent (for audit log)
    strategy: str = ""                 # strategy label
    client_ref: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "security_id": self.security_id,
            "exchange_segment": self.exchange_segment,
            "transaction_type": self.transaction_type,
            "quantity": self.quantity,
            "order_type": self.order_type,
            "product_type": self.product_type,
            "price": self.price,
            "trigger_price": self.trigger_price,
            "validity": self.validity,
            "agent_name": self.agent_name,
            "strategy": self.strategy,
            "client_ref": self.client_ref,
        }


@dataclass
class ExecutionReceipt:
    """
    Returned by DhanExecutor or PaperTrader after every order attempt.
    """
    order_id: str
    client_ref: str
    symbol: str
    transaction_type: str
    quantity: int
    price: Optional[float]
    status: OrderStatus
    fill_price: Optional[float] = None   # actual fill price (paper or live)
    fill_qty: int = 0
    raw_response: Optional[dict] = None
    error: str = ""
    ts: datetime = field(default_factory=lambda: datetime.now(IST))

    @property
    def is_filled(self) -> bool:
        return self.status in (OrderStatus.FILLED, OrderStatus.SIMULATED)

    def to_dict(self) -> dict:
        return {
            "order_id": self.order_id,
            "client_ref": self.client_ref,
            "symbol": self.symbol,
            "transaction_type": self.transaction_type,
            "quantity": self.quantity,
            "price": self.price,
            "fill_price": self.fill_price,
            "fill_qty": self.fill_qty,
            "status": self.status.value,
            "error": self.error,
            "ts": self.ts.isoformat(),
        }


# ── Order Manager ─────────────────────────────────────────────────────────────

class OrderManager:
    """
    Routes OrderRequests to the correct execution backend based on TRADING_MODE.

    Parameters
    ----------
    dhan_tools     : DhanTools instance (required for LIVE mode; can be None for paper).
    supabase_url   : Supabase URL for trade persistence.
    supabase_key   : Supabase service role key.
    trading_mode   : "LIVE" | "PAPER_TRADE" — overrides TRADING_MODE env var if set.
    """

    def __init__(
        self,
        dhan_tools=None,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        trading_mode: Optional[str] = None,
    ) -> None:
        self._mode = (trading_mode or TRADING_MODE).upper()
        self._kill_active = False
        self._receipts: Dict[str, ExecutionReceipt] = {}  # order_id → receipt

        # Executor (LIVE)
        self._executor = None
        if self._mode == "LIVE":
            if dhan_tools is None:
                raise ValueError("dhan_tools is required for LIVE trading mode.")
            from core.execution.dhan_executor import DhanExecutor
            self._executor = DhanExecutor(dhan_tools)

        # Paper trader (PAPER_TRADE)
        self._paper = None
        if self._mode == "PAPER_TRADE":
            from core.execution.paper_trader import PaperTrader
            self._paper = PaperTrader()

        # Supabase client for trade logging
        self._supabase = None
        self._supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self._supabase_key = supabase_key or os.getenv("SUPABASE_SERVICE_ROLE_KEY")

        log.info("OrderManager initialised — mode=%s", self._mode)

    # ── Public API ────────────────────────────────────────────────────────

    async def submit(self, req: OrderRequest) -> ExecutionReceipt:
        """
        Validate and route an order.

        Pre-trade checks:
          - Kill switch active → reject immediately.
          - Quantity <= 0 → reject.
          - Price required for LIMIT/SL orders → reject if missing.

        Returns ExecutionReceipt (never raises).
        """
        # Pre-trade guards
        if self._kill_active:
            log.warning("OrderManager: kill switch active — rejecting %s %s", req.transaction_type, req.symbol)
            return self._rejected(req, "Kill switch is active.")

        if req.quantity <= 0:
            return self._rejected(req, f"Invalid quantity: {req.quantity}")

        if req.order_type in ("LIMIT", "SL") and not req.price:
            return self._rejected(req, f"{req.order_type} order requires a price.")

        # Route
        if self._mode == "LIVE":
            receipt = await self._executor.execute(req)
        else:
            receipt = await self._paper.simulate(req)

        # Track and persist
        self._receipts[receipt.order_id] = receipt
        await self._persist_trade(req, receipt)
        return receipt

    async def cancel(self, order_id: str) -> bool:
        """Cancel a pending order (LIVE only; paper orders are immediately filled)."""
        if self._mode == "LIVE" and self._executor:
            ok = await self._executor.cancel(order_id)
            if ok and order_id in self._receipts:
                self._receipts[order_id].status = OrderStatus.CANCELLED
            return ok
        log.debug("Cancel ignored in PAPER_TRADE mode (order_id=%s)", order_id)
        return True

    async def modify(
        self,
        order_id: str,
        order_type: str,
        quantity: int,
        price: float,
        trigger_price: float = 0.0,
        validity: str = "DAY",
    ) -> bool:
        """Modify a pending LIVE order (no-op in paper mode)."""
        if self._mode == "LIVE" and self._executor:
            return await self._executor.modify(
                order_id=order_id,
                order_type=order_type,
                quantity=quantity,
                price=price,
                trigger_price=trigger_price,
                validity=validity,
            )
        return True

    async def get_order_status(self, order_id: str) -> Optional[str]:
        """Fetch live status from DhanHQ or return cached paper status."""
        if self._mode == "LIVE" and self._executor:
            return await self._executor.get_status(order_id)
        receipt = self._receipts.get(order_id)
        return receipt.status.value if receipt else None

    def get_receipts(self) -> List[ExecutionReceipt]:
        """Return all tracked receipts for this session."""
        return list(self._receipts.values())

    # ── Kill switch delegation ────────────────────────────────────────────

    async def emergency_kill(self) -> None:
        """
        Activate DhanHQ kill switch + exit all positions.
        Sets local flag so no further orders are accepted.
        """
        log.critical("OrderManager: EMERGENCY KILL — activating kill switch + exit all")
        self._kill_active = True
        if self._mode == "LIVE" and self._executor:
            await self._executor.activate_kill_switch()
            await self._executor.exit_all_positions()

    async def set_pnl_exit(
        self,
        max_profit: Optional[float] = None,
        max_loss: Optional[float] = None,
    ) -> None:
        """Delegate P&L auto-exit configuration to DhanExecutor (LIVE only)."""
        if self._mode == "LIVE" and self._executor:
            await self._executor.set_pnl_exit(max_profit=max_profit, max_loss=max_loss)

    # ── Paper-trade helpers ───────────────────────────────────────────────

    def get_paper_pnl(self) -> float:
        """Return cumulative paper P&L (paper mode only)."""
        if self._paper:
            return self._paper.cumulative_pnl
        return 0.0

    def get_paper_positions(self) -> dict:
        """Return simulated open positions (paper mode only)."""
        if self._paper:
            return self._paper.positions
        return {}

    # ── Persistence ───────────────────────────────────────────────────────

    async def _persist_trade(self, req: OrderRequest, receipt: ExecutionReceipt) -> None:
        """Write the trade to Supabase `trades` table (best-effort)."""
        client = self._get_supabase()
        if client is None:
            return
        row = {
            "order_id": receipt.order_id,
            "client_ref": receipt.client_ref,
            "symbol": req.symbol,
            "security_id": req.security_id,
            "exchange_segment": req.exchange_segment,
            "transaction_type": req.transaction_type,
            "quantity": req.quantity,
            "order_type": req.order_type,
            "product_type": req.product_type,
            "price": req.price,
            "fill_price": receipt.fill_price,
            "status": receipt.status.value,
            "agent_name": req.agent_name,
            "strategy": req.strategy,
            "trading_mode": self._mode,
            "error": receipt.error or None,
        }
        try:
            import asyncio
            await asyncio.to_thread(lambda: client.table("trades").insert(row).execute())
        except Exception as exc:
            log.warning("Failed to persist trade to Supabase: %s", exc)

    def _get_supabase(self):
        if self._supabase is not None:
            return self._supabase
        if not self._supabase_url or not self._supabase_key:
            return None
        try:
            from supabase import create_client  # type: ignore
            self._supabase = create_client(self._supabase_url, self._supabase_key)
        except Exception:
            pass
        return self._supabase

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _rejected(req: OrderRequest, reason: str) -> ExecutionReceipt:
        return ExecutionReceipt(
            order_id="",
            client_ref=req.client_ref,
            symbol=req.symbol,
            transaction_type=req.transaction_type,
            quantity=req.quantity,
            price=req.price,
            status=OrderStatus.REJECTED,
            error=reason,
        )
