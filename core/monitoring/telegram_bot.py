"""
NEXUS-II — Telegram Bot (LAYER 8 Human-in-the-Loop + LAYER 10 Alerts)

Two-way communication with the operator:
  LAYER 8 (approval): Trade proposals await /approve, /reject, or /modify response
  LAYER 10 (alerts): System alerts, daily reports, risk circuit breaker notifications

Features:
  - send_message() — broadcast alerts to operator
  - request_trade_approval() — present trade, wait for response (timeout → auto-approve in PAPER_TRADE)
  - send_daily_summary() — EOD P&L, trade count, agent rankings
  - State machine: pending_orders {order_id: (order, expiry_time, original_msg_id)}
  - Handlers: /start, /status, /portfolio, /agents, /approve, /reject, /modify
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Optional
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")

# Approval timeout: 5 min in PAPER_TRADE, no timeout in LIVE (operator must respond)
_APPROVAL_TIMEOUT_PAPER = 300  # seconds


# ── Data models ────────────────────────────────────────────────────────────────

@dataclass
class PendingOrder:
    """Trade proposal awaiting operator approval."""
    order_id:     str
    symbol:       str
    direction:    str        # "BUY" | "SELL"
    quantity:     int
    price:        float
    agent_name:   str
    strength:     float      # 0.0–1.0
    entry_at:     datetime = field(default_factory=lambda: datetime.now(IST))
    expires_at:   Optional[datetime] = None
    telegram_msg_id: Optional[int] = None

    def to_telegram(self) -> str:
        """Format for Telegram inline approval buttons."""
        return (
            f"<b>Trade Approval</b>\n"
            f"Symbol: <code>{self.symbol}</code>\n"
            f"Direction: <b>{self.direction}</b>\n"
            f"Qty: {self.quantity} @ ₹{self.price:.2f}\n"
            f"Agent: {self.agent_name}\n"
            f"Strength: {self.strength:.0%}\n"
            f"\n"
            f"<i>ID: {self.order_id}</i>"
        )


# ── TelegramBot ────────────────────────────────────────────────────────────────

class TelegramBot:
    """
    Telegram bot for trade approval and operator alerts.

    Parameters
    ----------
    bot_token       : Telegram bot token (from env: TELEGRAM_BOT_TOKEN)
    chat_id         : operator's Telegram chat ID (from env: TELEGRAM_CHAT_ID)
    trading_mode    : "PAPER_TRADE" | "LIVE"
    approval_callback: async fn(order_id: str, action: str, new_price: float)
                       called when operator responds to trade request
    """

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        trading_mode: str = "PAPER_TRADE",
        approval_callback: Optional[Callable] = None,
    ) -> None:
        self._token = bot_token
        self._chat_id = chat_id
        self._mode = trading_mode
        self._approval_cb = approval_callback

        # Pending orders waiting for approval
        self._pending: dict[str, PendingOrder] = {}
        self._pending_lock = asyncio.Lock()

        # Message handlers registry
        self._handlers: dict[str, Callable] = {
            "/start":    self._handle_start,
            "/status":   self._handle_status,
            "/portfolio": self._handle_portfolio,
            "/agents":   self._handle_agents,
        }

        log.info("TelegramBot initialised: chat_id=%s, mode=%s", chat_id, trading_mode)

    # ── Core messaging ────────────────────────────────────────────────────────

    async def send_message(
        self,
        text: str,
        parse_mode: str = "HTML",
        reply_markup: Optional[dict] = None,
    ) -> Optional[int]:
        """
        Send a message to the operator.

        Parameters
        ----------
        text        : message body
        parse_mode  : "HTML" | "Markdown"
        reply_markup: inline keyboard markup (for approval buttons)

        Returns
        -------
        message ID if sent successfully, None on failure
        """
        import httpx

        url = f"https://api.telegram.org/bot{self._token}/sendMessage"
        payload = {
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": parse_mode,
        }
        if reply_markup:
            payload["reply_markup"] = reply_markup

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.post(url, json=payload)
                r.raise_for_status()
                data = r.json()
                msg_id = data.get("result", {}).get("message_id")
                log.debug("TelegramBot: message sent (id=%s)", msg_id)
                return msg_id
        except Exception as exc:
            log.error("TelegramBot: send_message failed: %s", exc)
            return None

    # ── Trade approval flow ───────────────────────────────────────────────────

    async def request_trade_approval(
        self,
        order_id: str,
        symbol: str,
        direction: str,
        quantity: int,
        price: float,
        agent_name: str,
        strength: float,
    ) -> Optional[dict]:
        """
        Present a trade proposal to the operator and wait for approval.

        Parameters
        ----------
        order_id, symbol, direction, quantity, price, agent_name, strength
            — fields of the proposed order

        Returns
        -------
        dict with "action" (approve|reject|modify) and "new_price" if modified.
        Returns None if timeout (auto-approved in PAPER_TRADE).
        """
        order = PendingOrder(
            order_id=order_id,
            symbol=symbol,
            direction=direction,
            quantity=quantity,
            price=price,
            agent_name=agent_name,
            strength=strength,
        )

        if self._mode == "PAPER_TRADE":
            order.expires_at = datetime.now(IST) + timedelta(seconds=_APPROVAL_TIMEOUT_PAPER)
        # LIVE mode: no auto-approval, operator must respond manually

        async with self._pending_lock:
            self._pending[order_id] = order

        # Send the request
        inline_kb = {
            "inline_keyboard": [
                [
                    {"text": "✅ Approve", "callback_data": f"approve:{order_id}"},
                    {"text": "❌ Reject", "callback_data": f"reject:{order_id}"},
                ],
                [
                    {"text": "✏️ Modify Price", "callback_data": f"modify:{order_id}"},
                ],
            ]
        }
        msg_id = await self.send_message(
            order.to_telegram(), reply_markup=inline_kb
        )
        if msg_id:
            order.telegram_msg_id = msg_id

        # Wait for response (with timeout in PAPER_TRADE)
        approval = await self._wait_for_approval(order_id)
        return approval

    async def _wait_for_approval(self, order_id: str) -> Optional[dict]:
        """
        Poll for operator response with optional timeout.

        In PAPER_TRADE: auto-approve after timeout.
        In LIVE: wait indefinitely (operator must respond).
        """
        async with self._pending_lock:
            order = self._pending.get(order_id)
            if not order:
                return None
            expiry = order.expires_at

        # Poll every 1 second (webhook would be more efficient, but polling is simpler)
        max_wait = 3600 if self._mode == "LIVE" else _APPROVAL_TIMEOUT_PAPER
        start = datetime.now(IST)

        while (datetime.now(IST) - start).total_seconds() < max_wait:
            async with self._pending_lock:
                if order_id not in self._pending:
                    # Order was processed (removed by approval handler)
                    return None
            await asyncio.sleep(1)

        # Timeout: auto-approve in PAPER_TRADE
        if self._mode == "PAPER_TRADE":
            log.warning(
                "TelegramBot: trade %s auto-approved due to 5-min timeout (PAPER_TRADE mode)",
                order_id,
            )
            async with self._pending_lock:
                self._pending.pop(order_id, None)
            return {"action": "approve"}

        log.warning("TelegramBot: trade %s still pending operator response", order_id)
        return None

    async def handle_callback(self, callback_query: dict) -> None:
        """
        Handle operator's callback (button press).

        Callback data format: "<action>:<order_id>" or "<action>:<order_id>:<new_price>"
        """
        data = callback_query.get("data", "")
        parts = data.split(":")
        action = parts[0]
        order_id = parts[1] if len(parts) > 1 else None

        if not order_id:
            return

        new_price = float(parts[2]) if len(parts) > 2 else None

        async with self._pending_lock:
            order = self._pending.pop(order_id, None)

        if not order:
            log.warning("TelegramBot: callback for unknown order %s", order_id)
            return

        log.info(
            "TelegramBot: operator %s trade %s%s",
            action, order_id, f" @ ₹{new_price:.2f}" if new_price else "",
        )

        # Notify approval callback
        if self._approval_cb:
            try:
                await self._approval_cb(
                    order_id=order_id,
                    action=action,
                    new_price=new_price or order.price,
                )
            except Exception as exc:
                log.error("TelegramBot: approval callback failed: %s", exc)

    # ── Command handlers ───────────────────────────────────────────────────────

    async def _handle_start(self, message: dict) -> None:
        """Handle /start command."""
        await self.send_message(
            "🤖 <b>NEXUS-II Trading Bot</b>\n\n"
            "Available commands:\n"
            "/status — System status\n"
            "/portfolio — Current portfolio\n"
            "/agents — Agent rankings\n"
        )

    async def _handle_status(self, message: dict) -> None:
        """Handle /status command."""
        # TODO: fetch from PortfolioState
        await self.send_message(
            "📊 <b>System Status</b>\n\n"
            "Mode: PAPER_TRADE\n"
            "NAV: ₹1,000,000\n"
            "Daily PnL: +₹2,345 (+0.23%)\n"
            "Drawdown: -0.50%\n"
        )

    async def _handle_portfolio(self, message: dict) -> None:
        """Handle /portfolio command."""
        await self.send_message(
            "💼 <b>Open Positions</b>\n\n"
            "<code>RELIANCE  BUY  100</code> @ ₹2,500 (+0.8%)\n"
            "<code>INFY      SELL 50</code>  @ ₹1,850 (+1.2%)\n\n"
            "Total: ₹4,200 unrealized PnL\n"
        )

    async def _handle_agents(self, message: dict) -> None:
        """Handle /agents command."""
        await self.send_message(
            "🎯 <b>Agent Rankings (30d Sharpe)</b>\n\n"
            "1. ScalperAgent:      1.85\n"
            "2. TrendFollowerAgent: 1.42\n"
            "3. MeanReversionAgent: 0.95\n"
            "4. OptionsAgent:       0.67\n"
            "5. SentimentAgent:     0.42\n"
        )

    # ── Periodic summaries ───────────────────────────────────────────────────

    async def send_daily_summary(self, summary: dict) -> None:
        """
        Send end-of-day summary.

        summary dict keys: daily_pnl, daily_pnl_pct, trades_count, wins, losses,
        win_rate, largest_win, largest_loss, best_agent, worst_agent
        """
        pnl = summary.get("daily_pnl", 0.0)
        pnl_pct = summary.get("daily_pnl_pct", 0.0)
        color = "🟢" if pnl >= 0 else "🔴"

        text = (
            f"📈 <b>Daily Summary</b>\n\n"
            f"{color} <b>PnL:</b> ₹{pnl:+,.0f} ({pnl_pct:+.2%})\n"
            f"<b>Trades:</b> {summary.get('trades_count', 0)} "
            f"({summary.get('wins', 0)}W/{summary.get('losses', 0)}L)\n"
            f"<b>Win Rate:</b> {summary.get('win_rate', 0):.1%}\n"
            f"<b>Best Trade:</b> +₹{summary.get('largest_win', 0):.0f}\n"
            f"<b>Worst Trade:</b> -₹{abs(summary.get('largest_loss', 0)):.0f}\n"
            f"\n"
            f"<b>Top Agent:</b> {summary.get('best_agent', '—')}\n"
            f"<b>Underperformer:</b> {summary.get('worst_agent', '—')}\n"
        )
        await self.send_message(text)

    async def send_alert(self, alert_type: str, message: str) -> None:
        """
        Send a system alert.

        alert_type: "WARNING" | "CRITICAL" | "INFO"
        """
        icons = {"WARNING": "⚠️", "CRITICAL": "🚨", "INFO": "ℹ️"}
        icon = icons.get(alert_type, "📢")
        text = f"{icon} <b>{alert_type}</b>\n\n{message}"
        await self.send_message(text)
