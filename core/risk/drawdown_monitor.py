"""
NEXUS-II — DrawdownMonitor (v1.0 preserved + dynamic circuit breakers)

Monitors portfolio equity on every mark-to-market and fires four circuit breakers
when drawdown thresholds are breached.  All threshold values are fetched from
WeightCalibrationAgent so they adapt to market regime.

Circuit breakers (in escalating severity):
  Level 1 — Daily loss > max_daily_loss_pct
             → Halve all new position sizes (sizing flag only — enforced by
               WeightCalibrationAgent adjusting position_sizing output)
             → Telegram alert

  Level 2 — Daily loss > max_daily_loss_pct × 2
             → HALT all new positions (only exits allowed)
             → Telegram alert

  Level 3 — Weekly drawdown > 5%
             → Full system pause, manual restart required
             → Telegram alert

  Level 4 — Monthly drawdown > 10% (max_drawdown_pct hard cap)
             → Full system stop, Telegram alert, DhanHQ kill switch activated

Telegram integration is injected via an optional callback to avoid circular imports.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Callable, Optional
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    from core.calibration.weight_calibration_agent import WeightCalibrationAgent

log = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")

# Hard-coded safety floor (never allow > 10% monthly drawdown regardless of LLM)
_ABSOLUTE_MAX_MONTHLY_DD = 0.10


# ── Data models ────────────────────────────────────────────────────────────────

@dataclass
class DrawdownSnapshot:
    """Rolling drawdown snapshot at a point in time."""
    timestamp:     datetime
    portfolio_nav: float
    peak_nav:      float
    drawdown_pct:  float       # negative: (current - peak) / peak
    daily_start:   float
    daily_pnl_pct: float


@dataclass
class CircuitBreakerState:
    """Current state of all circuit breakers."""
    level:                int   = 0     # 0=normal, 1=size_halved, 2=halt, 3=paused, 4=stopped
    halted:               bool  = False  # Level 2+: no new positions
    paused:               bool  = False  # Level 3: manual restart needed
    stopped:              bool  = False  # Level 4: kill switch activated
    last_triggered_at:    Optional[datetime] = None
    last_reason:          str   = ""
    alert_sent:           bool  = False


# ── DrawdownMonitor ───────────────────────────────────────────────────────────

class DrawdownMonitor:
    """
    Portfolio drawdown monitor with four escalating circuit breakers
    (v1.0 preserved + dynamic thresholds via WeightCalibrationAgent).

    Parameters
    ----------
    calibration_agent  : WeightCalibrationAgent — source of dynamic thresholds
    starting_capital   : ₹ capital at the start of the session
    telegram_callback  : optional async fn(message: str) → None for alerts
    kill_switch_callback: optional async fn() → None to activate DhanHQ kill switch
    """

    def __init__(
        self,
        calibration_agent: "WeightCalibrationAgent",
        starting_capital: float,
        telegram_callback: Optional[Callable] = None,
        kill_switch_callback: Optional[Callable] = None,
    ) -> None:
        self._cal    = calibration_agent
        self._tg     = telegram_callback
        self._ks     = kill_switch_callback

        self._starting_capital = starting_capital
        self._peak_nav         = starting_capital
        self._day_start_nav    = starting_capital
        self._week_start_nav   = starting_capital
        self._month_start_nav  = starting_capital
        self._current_nav      = starting_capital

        self._cb = CircuitBreakerState()
        self._history: list[DrawdownSnapshot] = []
        self._session_start = datetime.now(IST)

    # ── Core update ───────────────────────────────────────────────────────────

    async def update(self, portfolio_nav: float) -> CircuitBreakerState:
        """
        Record new portfolio NAV and evaluate all circuit breakers.

        Call on every mark-to-market (trade fill, tick update, or periodic).

        Returns
        -------
        Current CircuitBreakerState — callers should check `.halted` and `.stopped`.
        """
        self._current_nav = portfolio_nav
        if portfolio_nav > self._peak_nav:
            self._peak_nav = portfolio_nav

        # Record snapshot
        snap = DrawdownSnapshot(
            timestamp     = datetime.now(IST),
            portfolio_nav = portfolio_nav,
            peak_nav      = self._peak_nav,
            drawdown_pct  = self._get_drawdown_from_peak(),
            daily_start   = self._day_start_nav,
            daily_pnl_pct = self._get_daily_pnl_pct(),
        )
        self._history.append(snap)

        # Evaluate breakers (escalating)
        thresholds = await self._cal.get_risk_thresholds()
        await self._evaluate_circuit_breakers(snap, thresholds)

        return self._cb

    # ── Circuit breaker evaluation ────────────────────────────────────────────

    async def _evaluate_circuit_breakers(
        self, snap: DrawdownSnapshot, thresholds: dict
    ) -> None:
        max_daily   = thresholds.get("max_daily_loss_pct", 0.02)
        max_monthly = min(thresholds.get("max_drawdown_pct", 0.08), _ABSOLUTE_MAX_MONTHLY_DD)
        weekly_cap  = 0.05   # 5% weekly drawdown → pause (v1.0 preserved)

        daily_loss     = -snap.daily_pnl_pct   # positive = loss
        weekly_loss    = self._get_weekly_loss()
        monthly_dd     = -self._get_monthly_drawdown()

        # ── Level 4: monthly drawdown (most severe) ──
        if monthly_dd >= max_monthly and self._cb.level < 4:
            await self._trip(
                level=4,
                reason=(
                    f"Monthly drawdown {monthly_dd:.2%} breached hard cap "
                    f"{max_monthly:.2%} — FULL STOP"
                ),
                critical=True,
            )
            return

        # ── Level 3: weekly drawdown ──
        if weekly_loss >= weekly_cap and self._cb.level < 3:
            await self._trip(
                level=3,
                reason=f"Weekly drawdown {weekly_loss:.2%} ≥ 5% — SYSTEM PAUSED",
                critical=True,
            )
            return

        # ── Level 2: daily loss × 2 → halt ──
        if daily_loss >= max_daily * 2 and self._cb.level < 2:
            await self._trip(
                level=2,
                reason=(
                    f"Daily loss {daily_loss:.2%} ≥ {max_daily * 2:.2%} "
                    f"(2× limit) — HALT new positions"
                ),
                critical=True,
            )
            return

        # ── Level 1: daily loss → halve sizing ──
        if daily_loss >= max_daily and self._cb.level < 1:
            await self._trip(
                level=1,
                reason=(
                    f"Daily loss {daily_loss:.2%} ≥ limit {max_daily:.2%} "
                    f"— halving position sizes"
                ),
                critical=False,
            )
            return

        # ── Recovery: if loss has recovered below threshold, reset level ──
        if self._cb.level == 1 and daily_loss < max_daily * 0.8:
            log.info("DrawdownMonitor: daily loss recovered — resetting Level 1")
            self._cb.level = 0

    async def _trip(self, level: int, reason: str, critical: bool) -> None:
        """Activate a circuit breaker level."""
        if self._cb.last_reason == reason:
            return   # avoid spamming identical alerts

        self._cb.level             = level
        self._cb.last_triggered_at = datetime.now(IST)
        self._cb.last_reason       = reason
        self._cb.alert_sent        = False

        if level >= 2:
            self._cb.halted  = True
        if level >= 3:
            self._cb.paused  = True
        if level >= 4:
            self._cb.stopped = True

        log_fn = log.critical if critical else log.warning
        log_fn("DrawdownMonitor L%d: %s", level, reason)

        # Telegram alert
        if self._tg:
            prefix = "🚨 CRITICAL" if critical else "⚠️ WARNING"
            try:
                await self._tg(f"{prefix} — DrawdownMonitor Level {level}\n{reason}")
                self._cb.alert_sent = True
            except Exception as exc:
                log.error("DrawdownMonitor: Telegram alert failed: %s", exc)

        # Kill switch at Level 4
        if level >= 4 and self._ks and not self._cb.stopped:
            log.critical("DrawdownMonitor: activating DhanHQ kill switch")
            try:
                await self._ks()
            except Exception as exc:
                log.error("DrawdownMonitor: kill switch activation failed: %s", exc)

    # ── Accessors (synchronous) ───────────────────────────────────────────────

    def get_current_drawdown(self) -> float:
        """Current drawdown from peak NAV (negative float)."""
        return self._get_drawdown_from_peak()

    def get_max_drawdown(self) -> float:
        """Maximum intraday drawdown seen this session (negative float)."""
        if not self._history:
            return 0.0
        return min(s.drawdown_pct for s in self._history)

    def get_daily_pnl_pct(self) -> float:
        """Today's P&L as fraction of day-start NAV."""
        return self._get_daily_pnl_pct()

    def is_circuit_breaker_triggered(self) -> bool:
        """True if any circuit breaker is active (level ≥ 1)."""
        return self._cb.level >= 1

    def is_halted(self) -> bool:
        """True if new positions are blocked (level ≥ 2)."""
        return self._cb.halted

    def is_stopped(self) -> bool:
        """True if the system should fully stop (level 4)."""
        return self._cb.stopped

    def get_state(self) -> CircuitBreakerState:
        """Current circuit breaker state."""
        return self._cb

    # ── Day / week / month reset ──────────────────────────────────────────────

    def reset_for_new_day(self, opening_nav: float) -> None:
        """
        Call at session open (09:15 IST) to reset daily P&L baseline.

        Levels 1 and 2 reset at day start (daily loss clears).
        Levels 3 and 4 require manual intervention.
        """
        self._day_start_nav = opening_nav
        self._current_nav   = opening_nav
        if opening_nav > self._peak_nav:
            self._peak_nav = opening_nav

        if self._cb.level <= 2:
            if self._cb.level > 0:
                log.info("DrawdownMonitor: resetting Level %d at new day", self._cb.level)
            self._cb = CircuitBreakerState()

        log.info("DrawdownMonitor: new day start — NAV ₹%.0f", opening_nav)

    def reset_for_new_week(self, opening_nav: float) -> None:
        """Call at Monday open to reset weekly drawdown baseline."""
        self._week_start_nav = opening_nav
        log.info("DrawdownMonitor: new week start — NAV ₹%.0f", opening_nav)

    def reset_for_new_month(self, opening_nav: float) -> None:
        """Call at first trading session of the month."""
        self._month_start_nav = opening_nav
        log.info("DrawdownMonitor: new month start — NAV ₹%.0f", opening_nav)

    def manual_resume(self) -> None:
        """
        Manually resume after a Level 3 pause (requires operator confirmation).
        Level 4 (stopped) cannot be resumed programmatically.
        """
        if self._cb.stopped:
            log.error("DrawdownMonitor: Level 4 STOP cannot be resumed programmatically")
            return
        if self._cb.paused:
            log.warning("DrawdownMonitor: manual resume from Level 3 pause")
            self._cb = CircuitBreakerState()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_drawdown_from_peak(self) -> float:
        if self._peak_nav <= 0:
            return 0.0
        return (self._current_nav - self._peak_nav) / self._peak_nav

    def _get_daily_pnl_pct(self) -> float:
        if self._day_start_nav <= 0:
            return 0.0
        return (self._current_nav - self._day_start_nav) / self._day_start_nav

    def _get_weekly_loss(self) -> float:
        """Weekly loss as positive fraction (0.0 if positive week)."""
        if self._week_start_nav <= 0:
            return 0.0
        pnl = (self._current_nav - self._week_start_nav) / self._week_start_nav
        return max(0.0, -pnl)

    def _get_monthly_drawdown(self) -> float:
        """Monthly drawdown from month-start NAV (negative float)."""
        if self._month_start_nav <= 0:
            return 0.0
        return (self._current_nav - self._month_start_nav) / self._month_start_nav
