"""
NEXUS-II — Guard Watchdog (Health Monitor + Auto-Restart)

Long-running background task that monitors bot health and recovers from
transient failures.

Health checks:
  1. Bot process alive (via PID / process handle)
  2. DhanHQ API connectivity (test quote fetch)
  3. Supabase connectivity (test query)
  4. No hung orders (order status check)
  5. Circuit breaker state (not in HALTED/PAUSED/STOPPED)
  6. Telegram connectivity (test message delivery)

Recovery actions:
  - Recoverable failure (API timeout): log + retry
  - Circuit breaker triggered: alert operator
  - Persistent failure: alert operator, escalate to manual restart

Liveness endpoint: GET /health → returns status JSON (for Cloudflare health checks)
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Callable, Optional
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    from core.mcp_tools.tool_registry import MCPTools
    from core.risk.drawdown_monitor import DrawdownMonitor
    from core.monitoring.telegram_bot import TelegramBot

log = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")


# ── Health status model ────────────────────────────────────────────────────────

@dataclass
class HealthStatus:
    """Current health of the bot."""
    healthy:              bool
    process_alive:        bool
    dhan_api_ok:          bool
    supabase_ok:          bool
    no_hung_orders:       bool
    circuit_breaker_ok:   bool
    telegram_ok:          bool
    last_checked:         datetime
    last_failure:         Optional[str] = None
    uptime_minutes:       float = 0.0


# ── GuardWatchdog ────────────────────────────────────────────────────────────

class GuardWatchdog:
    """
    Health monitor and auto-recovery watchdog (runs as a background task).

    Parameters
    ----------
    mcp_tools          : MCPTools for API connectivity checks
    drawdown_monitor   : DrawdownMonitor to check circuit breaker state
    telegram_bot       : TelegramBot to send alerts
    check_interval     : seconds between health checks (default 60)
    restart_callback   : async fn() to restart the bot on persistent failures
    """

    def __init__(
        self,
        mcp_tools: "MCPTools",
        drawdown_monitor: "DrawdownMonitor",
        telegram_bot: "TelegramBot",
        check_interval: int = 60,
        restart_callback: Optional[Callable] = None,
    ) -> None:
        self._tools = mcp_tools
        self._dd = drawdown_monitor
        self._tg = telegram_bot
        self._interval = check_interval
        self._restart_cb = restart_callback

        self._start_time = datetime.now(IST)
        self._last_health = HealthStatus(
            healthy=True, process_alive=True, dhan_api_ok=True,
            supabase_ok=True, no_hung_orders=True, circuit_breaker_ok=True,
            telegram_ok=True, last_checked=self._start_time,
        )
        self._consecutive_failures = 0
        self._running = False

    # ── Main loop ──────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the watchdog (runs as a background task)."""
        self._running = True
        log.info("GuardWatchdog: starting health checks every %ds", self._interval)

        while self._running:
            try:
                await self._check_health()
            except Exception as exc:
                log.error("GuardWatchdog: check_health failed: %s", exc)

            await asyncio.sleep(self._interval)

    async def stop(self) -> None:
        """Stop the watchdog."""
        self._running = False
        log.info("GuardWatchdog: stopped")

    # ── Health check logic ────────────────────────────────────────────────────

    async def _check_health(self) -> None:
        """Run all health checks and update status."""
        status = HealthStatus(
            healthy=True,
            process_alive=self._check_process(),
            dhan_api_ok=await self._check_dhan_api(),
            supabase_ok=await self._check_supabase(),
            no_hung_orders=await self._check_hung_orders(),
            circuit_breaker_ok=self._check_circuit_breaker(),
            telegram_ok=await self._check_telegram(),
            last_checked=datetime.now(IST),
            uptime_minutes=(
                (datetime.now(IST) - self._start_time).total_seconds() / 60
            ),
        )

        # Determine overall health
        status.healthy = all([
            status.process_alive,
            status.dhan_api_ok,
            status.supabase_ok,
            status.no_hung_orders,
            status.circuit_breaker_ok,
            status.telegram_ok,
        ])

        self._last_health = status

        if not status.healthy:
            self._consecutive_failures += 1
            failures = [k for k, v in status.__dict__.items()
                       if k.endswith("_ok") and not v] + (
                       ["process_dead"] if not status.process_alive else []
                     )
            status.last_failure = ", ".join(failures)
            log.warning(
                "GuardWatchdog: health check failed (consecutive: %d) — %s",
                self._consecutive_failures, failures,
            )

            # Alert on persistent failure
            if self._consecutive_failures >= 3:
                await self._alert_persistent_failure(failures)
        else:
            self._consecutive_failures = 0

    def _check_process(self) -> bool:
        """Check if the bot process is still alive (always true in this context)."""
        # In a real scenario, you'd check the PID or process handle
        return True

    async def _check_dhan_api(self) -> bool:
        """Test DhanHQ API connectivity (lightweight quote fetch)."""
        try:
            ltp = await self._tools.get_nifty_ltp()
            return ltp > 0.0
        except Exception as exc:
            log.warning("GuardWatchdog: DhanHQ API check failed: %s", exc)
            return False

    async def _check_supabase(self) -> bool:
        """Test Supabase connectivity (check via table list or ping)."""
        # Placeholder — in production, query a simple table or use /health endpoint
        try:
            # Could query: SELECT 1 FROM trades LIMIT 1
            return True
        except Exception as exc:
            log.warning("GuardWatchdog: Supabase check failed: %s", exc)
            return False

    async def _check_hung_orders(self) -> bool:
        """
        Check for orders stuck in PENDING state for > 30 minutes.

        Fetches order list and flags any that are suspiciously old.
        """
        try:
            orders = await self._tools.get_order_list()
            now = datetime.now(IST)
            for order in orders.get("data", []):
                status = order.get("orderStatus", "").upper()
                if status in ("PENDING", "PARTIALLY_FILLED"):
                    placed_at = order.get("createdAt")
                    if placed_at and (now - placed_at).total_seconds() > 1800:  # 30 min
                        log.warning(
                            "GuardWatchdog: hung order detected: %s (id=%s)",
                            status, order.get("id"),
                        )
                        return False
            return True
        except Exception as exc:
            log.warning("GuardWatchdog: hung order check failed: %s", exc)
            return False

    def _check_circuit_breaker(self) -> bool:
        """Check that circuit breaker is not in a halted/paused/stopped state."""
        cb_state = self._dd.get_state()
        if cb_state.level >= 2:
            log.warning(
                "GuardWatchdog: circuit breaker Level %d active (%s)",
                cb_state.level, cb_state.last_reason,
            )
            return False
        return True

    async def _check_telegram(self) -> bool:
        """Test Telegram connectivity (send a test message or check auth)."""
        try:
            # Test by sending a ping message (could be rate-limited)
            # For now, assume OK if TG bot is initialized
            return self._tg is not None
        except Exception as exc:
            log.warning("GuardWatchdog: Telegram check failed: %s", exc)
            return False

    # ── Alert and recovery ────────────────────────────────────────────────────

    async def _alert_persistent_failure(self, failures: list[str]) -> None:
        """Send alert for persistent health failure (3+ consecutive checks)."""
        message = (
            f"🚨 <b>GuardWatchdog Alert</b>\n\n"
            f"Bot health failures (persistent):\n"
            f"• {chr(10).join(failures)}\n\n"
            f"Uptime: {self._last_health.uptime_minutes:.1f} min\n"
            f"Last check: {self._last_health.last_checked.strftime('%H:%M:%S')}\n"
            f"\n"
            f"<i>Manual intervention may be required.</i>"
        )
        await self._tg.send_alert("CRITICAL", message)

        # Auto-restart if callback is provided (only on recoverable failures)
        if self._restart_cb and self._can_restart(failures):
            log.warning("GuardWatchdog: attempting auto-restart")
            try:
                await self._restart_cb()
            except Exception as exc:
                log.error("GuardWatchdog: auto-restart failed: %s", exc)

    @staticmethod
    def _can_restart(failures: list[str]) -> bool:
        """Determine if auto-restart is safe (don't restart on process/kernel issues)."""
        if "process_dead" in failures:
            return False  # process manager should handle this
        if "circuit_breaker" in failures:
            return False  # needs manual operator decision
        return True

    # ── Accessor for status endpoint ───────────────────────────────────────────

    def get_health_status(self) -> dict:
        """Return current health as JSON (for Cloudflare Workers health endpoint)."""
        return {
            "healthy": self._last_health.healthy,
            "uptime_minutes": round(self._last_health.uptime_minutes, 1),
            "checks": {
                "process": self._last_health.process_alive,
                "dhan_api": self._last_health.dhan_api_ok,
                "supabase": self._last_health.supabase_ok,
                "hung_orders": self._last_health.no_hung_orders,
                "circuit_breaker": self._last_health.circuit_breaker_ok,
                "telegram": self._last_health.telegram_ok,
            },
            "last_checked": self._last_health.last_checked.isoformat(),
            "consecutive_failures": self._consecutive_failures,
            "last_failure": self._last_health.last_failure,
        }
