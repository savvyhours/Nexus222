"""
NEXUS-II — Dhan Executor
Live-order execution layer wrapping DhanTools for LIVE trading mode.

Responsibilities:
  • Translate an OrderRequest into a DhanHQ API call (place / modify / cancel).
  • Confirm the order was accepted and return a standardised ExecutionReceipt.
  • Activate the kill switch and exit-all on system-level CRISIS signals.
  • All methods are async; SDK calls are delegated to DhanTools (already async).

This class NEVER runs in PAPER_TRADE mode — OrderManager routes there instead.

Usage:
    executor = DhanExecutor(dhan_tools)
    receipt  = await executor.execute(order_request)
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

from core.execution.order_manager import ExecutionReceipt, OrderRequest, OrderStatus

log = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")


class DhanExecutor:
    """
    Executes OrderRequests against the live DhanHQ v2 API.

    Parameters
    ----------
    dhan_tools : DhanTools instance (already authenticated).
    """

    def __init__(self, dhan_tools) -> None:
        self._dhan = dhan_tools

    # ── Order execution ───────────────────────────────────────────────────

    async def execute(self, req: OrderRequest) -> ExecutionReceipt:
        """
        Submit a new order to DhanHQ.

        Maps OrderRequest fields to DhanHQ place_order parameters.
        Returns ExecutionReceipt with order_id on acceptance.
        """
        log.info(
            "[LIVE] Placing %s %s × %d @ %s (%s)",
            req.transaction_type, req.symbol, req.quantity,
            req.price or "MARKET", req.product_type,
        )
        try:
            resp = await self._dhan.place_order(
                security_id=req.security_id,
                exchange_segment=req.exchange_segment,
                transaction_type=req.transaction_type,
                quantity=req.quantity,
                order_type=req.order_type,
                product_type=req.product_type,
                price=req.price or 0.0,
                trigger_price=req.trigger_price or 0.0,
                validity=req.validity,
                disclosed_quantity=0,
                after_market_order=req.amo,
                amo_time=req.amo_time,
                bo_profit_value=req.bo_profit_value,
                bo_stop_loss_value=req.bo_stop_loss_value,
            )
            order_id = self._extract_order_id(resp)
            status = OrderStatus.PENDING if order_id else OrderStatus.REJECTED
            return ExecutionReceipt(
                order_id=order_id or "",
                client_ref=req.client_ref,
                symbol=req.symbol,
                transaction_type=req.transaction_type,
                quantity=req.quantity,
                price=req.price,
                status=status,
                raw_response=resp,
            )
        except Exception as exc:
            log.error("[LIVE] place_order failed for %s: %s", req.symbol, exc)
            return ExecutionReceipt(
                order_id="",
                client_ref=req.client_ref,
                symbol=req.symbol,
                transaction_type=req.transaction_type,
                quantity=req.quantity,
                price=req.price,
                status=OrderStatus.REJECTED,
                error=str(exc),
            )

    async def modify(
        self,
        order_id: str,
        order_type: str,
        quantity: int,
        price: float,
        trigger_price: float = 0.0,
        validity: str = "DAY",
        leg_name: str = "",
    ) -> bool:
        """Modify a pending order. Returns True on success."""
        log.info("[LIVE] Modifying order %s → price=%.2f qty=%d", order_id, price, quantity)
        try:
            await self._dhan.modify_order(
                order_id=order_id,
                order_type=order_type,
                leg_name=leg_name,
                quantity=quantity,
                price=price,
                trigger_price=trigger_price,
                validity=validity,
            )
            return True
        except Exception as exc:
            log.error("[LIVE] modify_order %s failed: %s", order_id, exc)
            return False

    async def cancel(self, order_id: str) -> bool:
        """Cancel a pending order. Returns True on success."""
        log.info("[LIVE] Cancelling order %s", order_id)
        try:
            await self._dhan.cancel_order(order_id)
            return True
        except Exception as exc:
            log.error("[LIVE] cancel_order %s failed: %s", order_id, exc)
            return False

    async def get_status(self, order_id: str) -> Optional[str]:
        """Fetch current order status string from DhanHQ."""
        try:
            resp = await self._dhan.get_order_status(order_id)
            data = resp.get("data", {})
            return data.get("orderStatus", None)
        except Exception as exc:
            log.error("[LIVE] get_order_status %s failed: %s", order_id, exc)
            return None

    # ── Risk controls ─────────────────────────────────────────────────────

    async def activate_kill_switch(self) -> bool:
        """
        Activate DhanHQ Kill Switch.
        Blocks ALL new orders for the rest of the session.
        Called by DrawdownMonitor on CRISIS regime.
        IRREVERSIBLE until next trading session.
        """
        log.critical("[LIVE] *** KILL SWITCH ACTIVATED ***")
        try:
            await self._dhan.activate_kill_switch()
            return True
        except Exception as exc:
            log.critical("[LIVE] Kill switch activation FAILED: %s", exc)
            return False

    async def exit_all_positions(self) -> bool:
        """
        Exit all open positions at market price immediately.
        Called alongside kill switch in CRISIS / max-drawdown scenarios.
        """
        log.critical("[LIVE] *** EXIT ALL POSITIONS ***")
        try:
            await self._dhan.exit_all_positions()
            return True
        except Exception as exc:
            log.critical("[LIVE] exit_all_positions FAILED: %s", exc)
            return False

    async def set_pnl_exit(
        self,
        max_profit: Optional[float] = None,
        max_loss: Optional[float] = None,
    ) -> bool:
        """Configure DhanHQ P&L-based auto-exit threshold."""
        try:
            await self._dhan.set_pnl_exit(max_profit=max_profit, max_loss=max_loss)
            log.info("[LIVE] P&L exit set — profit=₹%s loss=₹%s", max_profit, max_loss)
            return True
        except Exception as exc:
            log.error("[LIVE] set_pnl_exit failed: %s", exc)
            return False

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _extract_order_id(resp: dict) -> Optional[str]:
        """Pull orderId from DhanHQ response regardless of nesting."""
        if not resp:
            return None
        data = resp.get("data") or resp
        if isinstance(data, dict):
            return data.get("orderId") or data.get("order_id")
        return None
