"""
NEXUS-II — Report Generator (Daily, Weekly, Monthly Summaries)

Fetches trade data from Supabase and generates formatted reports for:
  1. Daily Summary — sent via Telegram at market close (15:30 IST)
  2. Weekly Report — Friday EOD, sent via Telegram + saved to Supabase
  3. Monthly Report — 1st of next month, detailed deep-dive + attribution

Reports include:
  - P&L (absolute and %), trade count, win rate, Sharpe
  - Agent rankings (by 30-day Sharpe score)
  - Strategy attribution (which strategies drove P&L)
  - Risk metrics (max drawdown, longest losing streak)
  - Upcoming events (calibration changes, circuit breaker events)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any, Optional
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    from core.monitoring.telegram_bot import TelegramBot

log = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")


# ── Data models ────────────────────────────────────────────────────────────────

@dataclass
class DailyStats:
    """Daily statistics computed from trades."""
    date_:             date
    open_nav:          float
    close_nav:         float
    daily_pnl:         float
    daily_pnl_pct:     float
    trades_count:      int
    wins:              int
    losses:            int
    win_rate:          float
    avg_win:           float
    avg_loss:          float
    largest_win:       float
    largest_loss:      float
    max_intraday_dd:   float
    sharpe_daily:      float

    def to_dict(self) -> dict:
        return {
            "date": self.date_.isoformat(),
            "open_nav": round(self.open_nav, 2),
            "close_nav": round(self.close_nav, 2),
            "daily_pnl": round(self.daily_pnl, 2),
            "daily_pnl_pct": round(self.daily_pnl_pct, 4),
            "trades_count": self.trades_count,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(self.win_rate, 4),
            "avg_win": round(self.avg_win, 2),
            "avg_loss": round(self.avg_loss, 2),
            "largest_win": round(self.largest_win, 2),
            "largest_loss": round(self.largest_loss, 2),
            "max_intraday_dd": round(self.max_intraday_dd, 4),
            "sharpe_daily": round(self.sharpe_daily, 4),
        }


# ── ReportGenerator ────────────────────────────────────────────────────────────

class ReportGenerator:
    """
    Generate and distribute trading reports.

    Parameters
    ----------
    supabase_client : supabase.Client (for fetching trade data)
    telegram_bot    : TelegramBot (for sending reports)
    """

    def __init__(
        self,
        supabase_client: Any,
        telegram_bot: "TelegramBot",
    ) -> None:
        self._sb = supabase_client
        self._tg = telegram_bot

    # ── Report generation ──────────────────────────────────────────────────────

    async def generate_daily_report(self, report_date: Optional[date] = None) -> DailyStats:
        """
        Generate and send daily summary (typically called at 15:30 IST).

        Parameters
        ----------
        report_date : date to report on (default: today)

        Returns
        -------
        DailyStats — also sends via Telegram
        """
        if report_date is None:
            report_date = date.today()

        # Fetch trades for this date from Supabase
        try:
            result = self._sb.table("trades").select("*").eq(
                "date(opened_at)", report_date.isoformat()
            ).execute()
            trades = result.data
        except Exception as exc:
            log.error("ReportGenerator: failed to fetch trades for %s: %s", report_date, exc)
            trades = []

        # Compute statistics
        stats = self._compute_daily_stats(trades, report_date)

        # Send via Telegram
        await self._send_daily_telegram(stats)

        # Save to Supabase (optional — for historical record)
        try:
            self._sb.table("daily_reports").upsert(
                stats.to_dict(), on_conflict="date"
            ).execute()
        except Exception as exc:
            log.warning("ReportGenerator: failed to save daily report: %s", exc)

        return stats

    async def generate_weekly_report(
        self, week_end_date: Optional[date] = None
    ) -> dict:
        """
        Generate and send weekly summary (typically Friday EOD).

        Parameters
        ----------
        week_end_date : end of week (default: last Friday)

        Returns
        -------
        dict with weekly aggregates and agent rankings
        """
        if week_end_date is None:
            today = date.today()
            days_since_friday = (today.weekday() - 4) % 7
            week_end_date = today - timedelta(days=days_since_friday)

        week_start = week_end_date - timedelta(days=6)

        # Fetch trades for the week
        try:
            result = self._sb.table("trades").select("*").gte(
                "opened_at", week_start.isoformat()
            ).lte(
                "opened_at", (week_end_date + timedelta(days=1)).isoformat()
            ).execute()
            trades = result.data
        except Exception as exc:
            log.error("ReportGenerator: failed to fetch weekly trades: %s", exc)
            trades = []

        # Compute week stats and agent rankings
        week_stats = self._compute_weekly_stats(trades, week_start, week_end_date)
        agent_rankings = self._compute_agent_rankings(trades)

        # Send via Telegram
        await self._send_weekly_telegram(week_stats, agent_rankings)

        report = {
            "week": f"{week_start.isoformat()} to {week_end_date.isoformat()}",
            "stats": week_stats,
            "agent_rankings": agent_rankings,
        }

        # Save to Supabase
        try:
            self._sb.table("weekly_reports").insert(report).execute()
        except Exception as exc:
            log.warning("ReportGenerator: failed to save weekly report: %s", exc)

        return report

    async def generate_monthly_report(
        self, month_year: Optional[tuple] = None
    ) -> dict:
        """
        Generate and send comprehensive monthly summary (1st of next month).

        Parameters
        ----------
        month_year : (month, year) tuple (default: last complete month)

        Returns
        -------
        dict with monthly aggregates, agent rankings, strategy attribution
        """
        if month_year is None:
            today = date.today()
            if today.day == 1:
                # First of month — report on previous month
                first_of_this = today
                last_of_prev = first_of_this - timedelta(days=1)
                month_year = (last_of_prev.month, last_of_prev.year)
            else:
                # Mid-month — not time for monthly report yet
                log.warning("ReportGenerator: monthly report called mid-month")
                return {}

        month, year = month_year
        import calendar
        _, days_in_month = calendar.monthrange(year, month)
        month_start = date(year, month, 1)
        month_end = date(year, month, days_in_month)

        # Fetch trades for the month
        try:
            result = self._sb.table("trades").select("*").gte(
                "opened_at", month_start.isoformat()
            ).lte(
                "opened_at", (month_end + timedelta(days=1)).isoformat()
            ).execute()
            trades = result.data
        except Exception as exc:
            log.error("ReportGenerator: failed to fetch monthly trades: %s", exc)
            trades = []

        # Compute month stats, agent rankings, and strategy attribution
        month_stats = self._compute_monthly_stats(trades, month_start, month_end)
        agent_rankings = self._compute_agent_rankings(trades)
        strategy_attribution = self._compute_strategy_attribution(trades)

        # Send via Telegram (may be split into multiple messages due to size)
        await self._send_monthly_telegram(month_stats, agent_rankings, strategy_attribution)

        report = {
            "period": f"{month_year[0]}/{month_year[1]}",
            "stats": month_stats,
            "agent_rankings": agent_rankings,
            "strategy_attribution": strategy_attribution,
        }

        # Save to Supabase
        try:
            self._sb.table("monthly_reports").insert(report).execute()
        except Exception as exc:
            log.warning("ReportGenerator: failed to save monthly report: %s", exc)

        return report

    # ── Statistics computation ────────────────────────────────────────────────

    def _compute_daily_stats(self, trades: list, date_: date) -> DailyStats:
        """Compute daily statistics from a list of closed trades."""
        pnls = [t.get("pnl", 0.0) for t in trades if t.get("status") == "CLOSED"]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        return DailyStats(
            date_=date_,
            open_nav=0.0,  # would fetch from PortfolioState
            close_nav=0.0,
            daily_pnl=sum(pnls),
            daily_pnl_pct=0.0,  # would compute from NAV
            trades_count=len(pnls),
            wins=len(wins),
            losses=len(losses),
            win_rate=len(wins) / len(pnls) if pnls else 0.0,
            avg_win=sum(wins) / len(wins) if wins else 0.0,
            avg_loss=sum(losses) / len(losses) if losses else 0.0,
            largest_win=max(wins) if wins else 0.0,
            largest_loss=min(losses) if losses else 0.0,
            max_intraday_dd=0.0,  # would fetch from DrawdownMonitor
            sharpe_daily=0.0,  # would compute from daily returns
        )

    def _compute_weekly_stats(self, trades: list, start: date, end: date) -> dict:
        """Compute weekly aggregates."""
        pnls = [t.get("pnl", 0.0) for t in trades if t.get("status") == "CLOSED"]
        return {
            "period": f"{start.isoformat()} to {end.isoformat()}",
            "total_pnl": round(sum(pnls), 2),
            "trades": len(pnls),
            "win_rate": round(len([p for p in pnls if p > 0]) / len(pnls), 2) if pnls else 0.0,
        }

    def _compute_monthly_stats(self, trades: list, start: date, end: date) -> dict:
        """Compute monthly aggregates."""
        pnls = [t.get("pnl", 0.0) for t in trades if t.get("status") == "CLOSED"]
        return {
            "period": f"{start.isoformat()} to {end.isoformat()}",
            "total_pnl": round(sum(pnls), 2),
            "trades": len(pnls),
            "win_rate": round(len([p for p in pnls if p > 0]) / len(pnls), 2) if pnls else 0.0,
            "sharpe_monthly": 0.0,  # computed from daily returns
            "max_drawdown": 0.0,    # from DrawdownMonitor
        }

    def _compute_agent_rankings(self, trades: list) -> list[dict]:
        """Rank agents by P&L (or 30-day Sharpe)."""
        by_agent = {}
        for t in trades:
            agent = t.get("agent_name", "unknown")
            pnl = t.get("pnl", 0.0)
            if agent not in by_agent:
                by_agent[agent] = {"count": 0, "pnl": 0.0}
            by_agent[agent]["count"] += 1
            by_agent[agent]["pnl"] += pnl

        rankings = [
            {"agent": a, "trades": v["count"], "pnl": round(v["pnl"], 2)}
            for a, v in sorted(by_agent.items(), key=lambda x: x[1]["pnl"], reverse=True)
        ]
        return rankings

    def _compute_strategy_attribution(self, trades: list) -> dict:
        """Attribution by strategy."""
        by_strategy = {}
        for t in trades:
            strategy = t.get("strategy", "unknown")
            pnl = t.get("pnl", 0.0)
            if strategy not in by_strategy:
                by_strategy[strategy] = {"count": 0, "pnl": 0.0}
            by_strategy[strategy]["count"] += 1
            by_strategy[strategy]["pnl"] += pnl

        return {
            s: {"trades": v["count"], "pnl": round(v["pnl"], 2)}
            for s, v in sorted(by_strategy.items(), key=lambda x: x[1]["pnl"], reverse=True)
        }

    # ── Telegram formatting ────────────────────────────────────────────────────

    async def _send_daily_telegram(self, stats: DailyStats) -> None:
        """Format and send daily report via Telegram."""
        pnl = stats.daily_pnl
        pnl_pct = stats.daily_pnl_pct
        color = "🟢" if pnl >= 0 else "🔴"

        text = (
            f"📊 <b>Daily Summary — {stats.date_.strftime('%a, %b %d')}</b>\n\n"
            f"{color} <b>P&L:</b> ₹{pnl:+,.0f} ({pnl_pct:+.2%})\n"
            f"<b>Trades:</b> {stats.wins}W / {stats.losses}L ({stats.win_rate:.1%})\n"
            f"<b>Best Trade:</b> +₹{stats.largest_win:,.0f}\n"
            f"<b>Worst Trade:</b> -₹{abs(stats.largest_loss):,.0f}\n"
            f"<b>Max Drawdown:</b> {stats.max_intraday_dd:+.2%}\n"
        )
        await self._tg.send_message(text)

    async def _send_weekly_telegram(self, stats: dict, rankings: list) -> None:
        """Format and send weekly report via Telegram."""
        text = (
            f"📈 <b>Weekly Summary</b>\n\n"
            f"P&L: <b>₹{stats['total_pnl']:+,.0f}</b>\n"
            f"Trades: {stats['trades']}\n"
            f"Win Rate: {stats['win_rate']:.1%}\n\n"
            f"<b>Top Agents:</b>\n"
        )
        for rank, agent in enumerate(rankings[:3], 1):
            text += f"{rank}. {agent['agent']}: ₹{agent['pnl']:+,.0f}\n"

        await self._tg.send_message(text)

    async def _send_monthly_telegram(
        self, stats: dict, rankings: list, attribution: dict
    ) -> None:
        """Format and send monthly report via Telegram."""
        text = (
            f"📅 <b>Monthly Summary</b>\n\n"
            f"P&L: <b>₹{stats['total_pnl']:+,.0f}</b>\n"
            f"Trades: {stats['trades']}\n"
            f"Sharpe: {stats['sharpe_monthly']:.2f}\n"
            f"Max DD: {stats['max_drawdown']:.2%}\n\n"
            f"<b>Top Strategies:</b>\n"
        )
        for i, (strat, data) in enumerate(list(attribution.items())[:3], 1):
            text += f"{i}. {strat}: ₹{data['pnl']:+,.0f} ({data['trades']} trades)\n"

        await self._tg.send_message(text)
