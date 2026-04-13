"""
NEXUS-II — MCP Tool Registry

Composes DhanTools, DataTools, and ComputeTools into a single `MCPTools`
facade consumed by every component in the system.

Also owns `PortfolioState` — the in-process portfolio ledger that tracks:
  - Starting capital and current NAV
  - Open positions (by symbol)
  - Daily P&L and peak NAV (for drawdown calculation)
  - Agent Sharpe-score cache (updated after each trade closure)

`MCPTools` is constructed once at startup and injected into:
  - WeightCalibrationAgent._gather_market_state()
  - BaseAgent subclasses (self._tools)
  - PreTradeRiskChecker
  - ExecutionEngine
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# PortfolioState — in-process portfolio ledger
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PortfolioState:
    """
    Lightweight in-process portfolio tracker.

    Updated by the ExecutionEngine after each fill / trade closure.
    Provides synchronous reads for drawdown / daily P&L so that
    WeightCalibrationAgent._gather_market_state() never blocks.

    Parameters
    ----------
    starting_capital : ₹ capital at the start of the trading day
    """

    starting_capital: float
    current_nav:      float = field(init=False)   # updated on every mark-to-market
    peak_nav:         float = field(init=False)   # high watermark
    day_start_nav:    float = field(init=False)   # for daily P&L %
    positions:        dict  = field(default_factory=dict)  # symbol → Position dict
    closed_trades:    list  = field(default_factory=list)

    # Per-agent rolling Sharpe cache: agent_key → float
    agent_sharpe_30d: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.current_nav   = self.starting_capital
        self.peak_nav      = self.starting_capital
        self.day_start_nav = self.starting_capital

    # ── Drawdown & P&L accessors (synchronous — no I/O) ──────────────────────

    def get_current_drawdown(self) -> float:
        """
        Current drawdown from peak NAV.

        Returns a negative float (e.g. -0.03 = 3% drawdown).
        Returns 0.0 if at or above peak.
        """
        if self.peak_nav <= 0:
            return 0.0
        return (self.current_nav - self.peak_nav) / self.peak_nav

    def get_daily_pnl_pct(self) -> float:
        """
        P&L as a fraction of day-start NAV.

        Returns a signed float (e.g. +0.012 = +1.2% today).
        """
        if self.day_start_nav <= 0:
            return 0.0
        return (self.current_nav - self.day_start_nav) / self.day_start_nav

    def get_agent_sharpe_scores(self) -> dict[str, float]:
        """
        Latest cached 30-day Sharpe scores for all agents.

        Returns empty dict if not yet populated (new agents treated as equal weight).
        """
        return dict(self.agent_sharpe_30d)

    # ── Mutators (called by ExecutionEngine) ─────────────────────────────────

    def mark_to_market(self, new_nav: float) -> None:
        """Update current NAV and recalculate peak/drawdown."""
        self.current_nav = new_nav
        if new_nav > self.peak_nav:
            self.peak_nav = new_nav

    def open_position(self, symbol: str, position: dict) -> None:
        """Register a new open position."""
        self.positions[symbol] = position
        log.debug("PortfolioState: opened %s", symbol)

    def close_position(self, symbol: str, pnl: float) -> None:
        """Remove position, record the trade, update NAV."""
        position = self.positions.pop(symbol, None)
        if position:
            self.closed_trades.append({**position, "pnl": pnl, "closed_at": time.time()})
        self.current_nav += pnl
        if self.current_nav > self.peak_nav:
            self.peak_nav = self.current_nav

    def update_agent_sharpe(self, agent_key: str, sharpe: float) -> None:
        """Update cached Sharpe score for an agent (called by monitoring layer)."""
        self.agent_sharpe_30d[agent_key] = sharpe

    def reset_for_new_day(self, opening_nav: float) -> None:
        """Call at the start of each trading session to reset daily P&L baseline."""
        self.day_start_nav = opening_nav
        self.current_nav   = opening_nav
        if opening_nav > self.peak_nav:
            self.peak_nav = opening_nav
        log.info("PortfolioState: new day — starting NAV ₹%.0f", opening_nav)


# ═══════════════════════════════════════════════════════════════════════════════
# MCPTools — unified tool facade
# ═══════════════════════════════════════════════════════════════════════════════

class MCPTools:
    """
    Single interface for all MCP tool calls throughout NEXUS-II.

    Exposes every method called by WeightCalibrationAgent._gather_market_state():
        get_india_vix()
        get_nifty_change()
        get_fii_dii()
        get_sector_momentum()
        get_advance_decline_ratio()
        get_avg_iv_percentile()
        get_current_drawdown()       ← sync (reads PortfolioState)
        get_daily_pnl_pct()          ← sync (reads PortfolioState)
        get_agent_sharpe_scores()    ← sync (reads PortfolioState)

    Also exposes all DhanTools market data and execution methods for agents
    and the execution engine.

    Parameters
    ----------
    dhan_tools     : DhanTools instance
    data_tools     : DataTools instance
    compute_tools  : module reference (or None — module is imported directly)
    portfolio_state: PortfolioState instance (shared with ExecutionEngine)
    """

    def __init__(
        self,
        dhan_tools: Any,
        data_tools: Any,
        portfolio_state: PortfolioState,
        compute_tools_module: Any = None,
    ) -> None:
        self.dhan     = dhan_tools
        self.data     = data_tools
        self.portfolio = portfolio_state
        # compute_tools is imported as a module so callers can use:
        # self._tools.compute.compute_all_indicators(df)
        if compute_tools_module is None:
            from core.mcp_tools import compute_tools as _ct
            self.compute = _ct
        else:
            self.compute = compute_tools_module

    # ═══════════════════════════════════════════════════════════════════════════
    # WeightCalibrationAgent._gather_market_state() interface
    # ═══════════════════════════════════════════════════════════════════════════

    async def get_india_vix(self) -> float:
        """Current India VIX. Falls back to 16.0."""
        return await self.data.get_india_vix()

    async def get_nifty_change(self) -> float:
        """Nifty 50 intraday % change. Falls back to 0.0."""
        return await self.data.get_nifty_change_pct()

    async def get_fii_dii(self) -> dict:
        """FII / DII flow summary (today + 3d + 5d + consecutive days)."""
        return await self.data.get_fii_dii_flows()

    async def get_sector_momentum(self) -> dict[str, float]:
        """NSE sector 5-day returns dict. Falls back to all-zeros."""
        return await self.data.get_sector_momentum()

    async def get_advance_decline_ratio(self) -> float:
        """NSE advance/decline ratio. Falls back to 1.0 (neutral)."""
        return await self.data.get_advance_decline_ratio()

    async def get_avg_iv_percentile(self) -> float:
        """Average NIFTY options IV percentile (0–100). Falls back to 50.0."""
        return await self.data.get_avg_iv_percentile(self.dhan)

    # Synchronous reads from PortfolioState (no await needed) ─────────────────

    def get_current_drawdown(self) -> float:
        """Current portfolio drawdown from peak (negative float). Sync."""
        return self.portfolio.get_current_drawdown()

    def get_daily_pnl_pct(self) -> float:
        """Today's P&L as fraction of starting NAV. Sync."""
        return self.portfolio.get_daily_pnl_pct()

    def get_agent_sharpe_scores(self) -> dict[str, float]:
        """Latest 30-day Sharpe scores for all agents. Sync."""
        return self.portfolio.get_agent_sharpe_scores()

    # ═══════════════════════════════════════════════════════════════════════════
    # Market data pass-throughs (DhanTools)
    # ═══════════════════════════════════════════════════════════════════════════

    async def get_quote(self, securities: dict) -> dict:
        return await self.dhan.get_quote(securities)

    async def get_ohlc(self, securities: dict) -> dict:
        return await self.dhan.get_ohlc(securities)

    async def get_market_depth(self, exchange: str, security_id: str, instrument_type: str = "EQUITY") -> dict:
        return await self.dhan.get_market_depth(exchange, security_id, instrument_type)

    async def get_intraday_bars(self, security_id: str, exchange_segment: str, instrument_type: str = "EQUITY") -> dict:
        return await self.dhan.get_intraday_bars(security_id, exchange_segment, instrument_type)

    async def get_historical_bars(self, security_id: str, exchange_segment: str, instrument_type: str, from_date: str, to_date: str, expiry_date: str = "") -> dict:
        return await self.dhan.get_historical_bars(security_id, exchange_segment, instrument_type, from_date, to_date, expiry_date)

    async def get_options_chain(self, under_security_id: str, under_exchange_segment: str, expiry: str) -> dict:
        return await self.dhan.get_options_chain(under_security_id, under_exchange_segment, expiry)

    async def get_positions(self) -> dict:
        return await self.dhan.get_positions()

    async def get_holdings(self) -> dict:
        return await self.dhan.get_holdings()

    # ═══════════════════════════════════════════════════════════════════════════
    # Execution pass-throughs (DhanTools)
    # ═══════════════════════════════════════════════════════════════════════════

    async def place_order(self, **kwargs) -> dict:
        return await self.dhan.place_order(**kwargs)

    async def place_super_order(self, **kwargs) -> dict:
        return await self.dhan.place_super_order(**kwargs)

    async def modify_order(self, **kwargs) -> dict:
        return await self.dhan.modify_order(**kwargs)

    async def cancel_order(self, order_id: str) -> dict:
        return await self.dhan.cancel_order(order_id)

    async def get_order_list(self) -> dict:
        return await self.dhan.get_order_list()

    async def get_order_status(self, order_id: str) -> dict:
        return await self.dhan.get_order_status(order_id)

    # ═══════════════════════════════════════════════════════════════════════════
    # Risk control pass-throughs (DhanTools)
    # ═══════════════════════════════════════════════════════════════════════════

    async def activate_kill_switch(self) -> dict:
        return await self.dhan.activate_kill_switch()

    async def set_pnl_exit(self, max_profit: Optional[float] = None, max_loss: Optional[float] = None) -> dict:
        return await self.dhan.set_pnl_exit(max_profit=max_profit, max_loss=max_loss)

    async def exit_all_positions(self) -> dict:
        return await self.dhan.exit_all_positions()

    # ═══════════════════════════════════════════════════════════════════════════
    # External data pass-throughs (DataTools)
    # ═══════════════════════════════════════════════════════════════════════════

    async def get_news_headlines(self, symbol: str, count: int = 20) -> list[str]:
        return await self.data.get_news_headlines(symbol, count)

    async def get_screener_fundamentals(self, symbol: str) -> dict:
        return await self.data.get_screener_fundamentals(symbol)

    async def get_rbi_events(self, lookahead_days: int = 7) -> list[str]:
        return await self.data.get_rbi_events(lookahead_days)

    async def get_usd_inr(self) -> float:
        return await self.data.get_usd_inr()

    async def get_brent_crude(self) -> float:
        return await self.data.get_brent_crude()

    # ── FRED macro data pass-throughs ─────────────────────────────────────────

    async def get_fed_funds_rate(self) -> float:
        """US Federal Funds Rate (%). FRED FEDFUNDS series."""
        return await self.data.get_fed_funds_rate()

    async def get_us_cpi_yoy(self) -> float:
        """US CPI year-over-year inflation (%). FRED CPIAUCSL series."""
        return await self.data.get_us_cpi_yoy()

    async def get_us_10y_yield(self) -> float:
        """US 10-Year Treasury yield (%). FRED DGS10 series."""
        return await self.data.get_us_10y_yield()

    async def get_dxy(self) -> float:
        """US Dollar Index (trade-weighted). FRED DTWEXBGS series."""
        return await self.data.get_dxy()

    async def get_us_macro_snapshot(self) -> dict:
        """All US macro indicators in one concurrent fetch (Fed rate, CPI, 10Y, DXY)."""
        return await self.data.get_us_macro_snapshot()

    # ── Finnhub data pass-throughs ────────────────────────────────────────────

    async def get_company_news_finnhub(self, symbol: str, count: int = 20) -> list[str]:
        """Company news headlines from Finnhub (higher quality than Yahoo RSS)."""
        return await self.data.get_company_news_finnhub(symbol, count)

    async def get_earnings_calendar(self, symbol: str) -> list[dict]:
        """Upcoming earnings dates and estimates for a symbol (Finnhub)."""
        return await self.data.get_earnings_calendar(symbol)

    async def get_economic_calendar(self) -> list[dict]:
        """Global high-impact economic events from Finnhub (US/IN/EU, medium+high)."""
        return await self.data.get_economic_calendar()

    async def get_market_sentiment_finnhub(self) -> dict:
        """India market sentiment proxy (bullish/bearish %) via Finnhub INDA ETF."""
        return await self.data.get_market_sentiment_finnhub()


# ═══════════════════════════════════════════════════════════════════════════════
# Factory function
# ═══════════════════════════════════════════════════════════════════════════════

def create_mcp_tools(
    dhan_client: Any,
    starting_capital: float,
    alpha_vantage_key: str = "",
    nse_session_cookie: str = "",
    fred_api_key: str = "",
    finnhub_api_key: str = "",
) -> MCPTools:
    """
    Factory: construct and wire together all MCP tool instances.

    Call this once at system startup and inject the returned MCPTools into
    WeightCalibrationAgent, agents, execution engine, and risk checker.

    Parameters
    ----------
    dhan_client       : initialised dhanhq.DhanHQ instance
    starting_capital  : trading capital in ₹ (e.g. 500_000)
    alpha_vantage_key : AlphaVantage API key for FOREX / commodity data
    nse_session_cookie: optional NSE cookie for advanced breadth endpoints
    fred_api_key      : FRED API key — US macro (Fed rate, CPI, 10Y, DXY)
    finnhub_api_key   : Finnhub API key — news, earnings, economic calendar
    """
    from core.mcp_tools.dhan_tools import DhanTools
    from core.mcp_tools.data_tools  import DataTools

    dhan_tools     = DhanTools(dhan_client)
    data_tools     = DataTools(
        alpha_vantage_key=alpha_vantage_key,
        nse_session_cookie=nse_session_cookie,
        fred_api_key=fred_api_key,
        finnhub_api_key=finnhub_api_key,
    )
    portfolio_state = PortfolioState(starting_capital=starting_capital)

    return MCPTools(
        dhan_tools=dhan_tools,
        data_tools=data_tools,
        portfolio_state=portfolio_state,
    )
