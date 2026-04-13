"""
NEXUS-II — DhanHQ v2 Tool Wrapper

Async wrappers around every DhanHQ v2 SDK call used by the system.
Rate limits enforced via asyncio.Semaphore + a dedicated token-bucket for
the options chain (1 call per 150 seconds — corrected from v2.0).

Sections:
  A. Market Data  — quote, OHLC, intraday, historical, depth, options chain
  B. Execution    — place / modify / cancel order, order status, positions
  C. Risk Controls — kill switch, P&L exit, exit all positions

All I/O methods are async. The DhanHQ SDK is synchronous, so calls are run
inside asyncio.get_event_loop().run_in_executor() to avoid blocking the event
loop.
"""
from __future__ import annotations

import asyncio
import logging
import time
from functools import partial
from typing import Any, Optional

log = logging.getLogger(__name__)

# ── Rate-limit constants ───────────────────────────────────────────────────────
_QUOTE_SEMAPHORE_LIMIT     = 1    # 1 call/sec for quote/ohlc/depth
_HIST_SEMAPHORE_LIMIT      = 5    # 5 calls/sec for historical/intraday
_ORDER_SEMAPHORE_LIMIT     = 10   # 10 calls/sec for order APIs
_OPTIONS_CHAIN_INTERVAL    = 150  # 1 call per 150 seconds (hard limit)


class _TokenBucket:
    """Simple token-bucket for options chain rate limiting."""

    def __init__(self, interval_seconds: float) -> None:
        self._interval = interval_seconds
        self._last_call: float = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self._interval - (now - self._last_call)
            if wait > 0:
                log.debug("OptionsChain rate-limit: waiting %.1fs", wait)
                await asyncio.sleep(wait)
            self._last_call = time.monotonic()


class DhanTools:
    """
    Async DhanHQ v2 API wrapper.

    Parameters
    ----------
    dhan_client : dhanhq.DhanHQ instance already initialised with access token
                  and client_id from environment variables.
    loop        : asyncio event loop (default: running loop)
    """

    def __init__(self, dhan_client: Any, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        self._dhan = dhan_client
        self._loop = loop or asyncio.get_event_loop()
        self._quote_sem = asyncio.Semaphore(_QUOTE_SEMAPHORE_LIMIT)
        self._hist_sem  = asyncio.Semaphore(_HIST_SEMAPHORE_LIMIT)
        self._order_sem = asyncio.Semaphore(_ORDER_SEMAPHORE_LIMIT)
        self._options_bucket = _TokenBucket(_OPTIONS_CHAIN_INTERVAL)

    # ── Internal executor helper ───────────────────────────────────────────────

    async def _run(self, func, *args, **kwargs) -> Any:
        """Run a blocking SDK call in the default thread-pool executor."""
        return await self._loop.run_in_executor(None, partial(func, *args, **kwargs))

    # ═══════════════════════════════════════════════════════════════════════════
    # A. MARKET DATA
    # ═══════════════════════════════════════════════════════════════════════════

    async def get_quote(self, securities: dict) -> dict:
        """
        LTP + OHLC snapshot for up to 1,000 instruments.

        Parameters
        ----------
        securities : dict mapping exchange_segment → list of security_ids
                     e.g. {"NSE_EQ": ["1333", "11536"]}

        Returns
        -------
        DhanHQ response dict with 'data' key containing instrument quotes.
        """
        async with self._quote_sem:
            return await self._run(self._dhan.get_ltp, securities)

    async def get_ohlc(self, securities: dict) -> dict:
        """OHLC snapshot for listed securities (same format as get_quote)."""
        async with self._quote_sem:
            return await self._run(self._dhan.ohlc_data, securities)

    async def get_market_depth(
        self,
        exchange: str,
        security_id: str,
        instrument_type: str = "EQUITY",
    ) -> dict:
        """
        Level-2 market depth (best 5 bid / ask).

        Parameters
        ----------
        exchange        : "NSE" | "BSE"
        security_id     : DhanHQ internal security id (str)
        instrument_type : "EQUITY" | "FUTIDX" | "OPTIDX" | ...
        """
        async with self._quote_sem:
            return await self._run(
                self._dhan.market_depth, exchange, security_id, instrument_type
            )

    async def get_intraday_bars(
        self,
        security_id: str,
        exchange_segment: str,
        instrument_type: str = "EQUITY",
    ) -> dict:
        """
        1-minute intraday OHLCV bars for the current trading session.

        Parameters
        ----------
        security_id       : DhanHQ security id
        exchange_segment  : "NSE_EQ" | "BSE_EQ" | "NSE_FNO" | ...
        instrument_type   : "EQUITY" | "FUTIDX" | "OPTIDX" | ...
        """
        async with self._hist_sem:
            return await self._run(
                self._dhan.intraday_minute_data,
                security_id,
                exchange_segment,
                instrument_type,
            )

    async def get_historical_bars(
        self,
        security_id: str,
        exchange_segment: str,
        instrument_type: str,
        from_date: str,
        to_date: str,
        expiry_date: str = "",
    ) -> dict:
        """
        Daily historical OHLCV bars.

        Parameters
        ----------
        from_date / to_date : "YYYY-MM-DD" strings
        expiry_date         : required for options/futures, blank for equity
        """
        async with self._hist_sem:
            return await self._run(
                self._dhan.historical_daily_data,
                security_id,
                exchange_segment,
                expiry_date,
                instrument_type,
                from_date,
                to_date,
            )

    async def get_options_chain(
        self,
        under_security_id: str,
        under_exchange_segment: str,
        expiry: str,
    ) -> dict:
        """
        Full options chain for an underlying at a given expiry.

        Rate-limited to 1 call per 150 seconds (Dhan hard limit).

        Parameters
        ----------
        under_security_id      : security id of the underlying (e.g. "13" for NIFTY)
        under_exchange_segment : "NSE_FNO"
        expiry                 : expiry date string "YYYY-MM-DD"
        """
        await self._options_bucket.acquire()
        return await self._run(
            self._dhan.option_chain,
            under_security_id,
            under_exchange_segment,
            expiry,
        )

    async def get_expiry_list(
        self, under_security_id: str, exchange_segment: str
    ) -> dict:
        """Available expiry dates for an underlying."""
        async with self._hist_sem:
            return await self._run(
                self._dhan.expiry_list, under_security_id, exchange_segment
            )

    async def get_positions(self) -> dict:
        """Current open positions (intraday + overnight)."""
        async with self._order_sem:
            return await self._run(self._dhan.get_positions)

    async def get_holdings(self) -> dict:
        """Overnight holdings (delivery positions)."""
        async with self._order_sem:
            return await self._run(self._dhan.get_holdings)

    # ═══════════════════════════════════════════════════════════════════════════
    # B. EXECUTION
    # ═══════════════════════════════════════════════════════════════════════════

    async def place_order(
        self,
        security_id: str,
        exchange_segment: str,
        transaction_type: str,       # "BUY" | "SELL"
        quantity: int,
        order_type: str,             # "MARKET" | "LIMIT" | "STOP_LOSS" | "STOP_LOSS_MARKET"
        product_type: str,           # "INTRADAY" | "CNC" | "MARGIN" | "MTF" | "CO" | "BO"
        price: float = 0.0,
        trigger_price: float = 0.0,
        disclosed_quantity: int = 0,
        after_market_order: bool = False,
        validity: str = "DAY",
        amo_time: str = "OPEN",
        bo_profit_value: float = 0.0,
        bo_stop_loss_value: float = 0.0,
        tag: Optional[str] = None,
    ) -> dict:
        """
        Place a regular order on DhanHQ.

        Returns the raw DhanHQ response dict containing order_id on success.
        Raises on HTTP errors (caller should catch).
        """
        async with self._order_sem:
            return await self._run(
                self._dhan.place_order,
                security_id=security_id,
                exchange_segment=exchange_segment,
                transaction_type=transaction_type,
                quantity=quantity,
                order_type=order_type,
                product_type=product_type,
                price=price,
                trigger_price=trigger_price,
                disclosed_quantity=disclosed_quantity,
                after_market_order=after_market_order,
                validity=validity,
                amo_time=amo_time,
                bo_profit_value=bo_profit_value,
                bo_stop_loss_value=bo_stop_loss_value,
                tag=tag,
            )

    async def place_super_order(
        self,
        security_id: str,
        exchange_segment: str,
        transaction_type: str,
        quantity: int,
        price: float,
        target: float,
        stop_loss: float,
        trailing_jump: float = 0.0,
        product_type: str = "CNC",
    ) -> dict:
        """
        Place a DhanHQ Super Order (bracket order with target + SL + optional trailing stop).

        trailing_jump > 0 enables trailing stop (₹ increment per step).
        """
        async with self._order_sem:
            return await self._run(
                self._dhan.place_super_order,
                security_id=security_id,
                exchange_segment=exchange_segment,
                transaction_type=transaction_type,
                quantity=quantity,
                price=price,
                target=target,
                stop_loss=stop_loss,
                trailing_jump=trailing_jump,
                product_type=product_type,
            )

    async def modify_order(
        self,
        order_id: str,
        order_type: str,
        leg_name: str,
        quantity: int,
        price: float,
        trigger_price: float,
        disclosed_quantity: int,
        validity: str,
    ) -> dict:
        """Modify a pending order."""
        async with self._order_sem:
            return await self._run(
                self._dhan.modify_order,
                order_id=order_id,
                order_type=order_type,
                leg_name=leg_name,
                quantity=quantity,
                price=price,
                trigger_price=trigger_price,
                disclosed_quantity=disclosed_quantity,
                validity=validity,
            )

    async def cancel_order(self, order_id: str) -> dict:
        """Cancel a pending order."""
        async with self._order_sem:
            return await self._run(self._dhan.cancel_order, order_id)

    async def get_order_list(self) -> dict:
        """All orders for the current trading session."""
        async with self._order_sem:
            return await self._run(self._dhan.get_order_list)

    async def get_order_status(self, order_id: str) -> dict:
        """Status of a specific order."""
        async with self._order_sem:
            return await self._run(self._dhan.get_order_by_id, order_id)

    async def get_trade_history(self, from_date: str, to_date: str, page: int = 0) -> dict:
        """Executed trade history between two dates ("YYYY-MM-DD")."""
        async with self._order_sem:
            return await self._run(
                self._dhan.get_trade_history, from_date, to_date, page
            )

    # ═══════════════════════════════════════════════════════════════════════════
    # C. RISK CONTROLS (DhanHQ Native)
    # ═══════════════════════════════════════════════════════════════════════════

    async def activate_kill_switch(self) -> dict:
        """
        Activate DhanHQ Kill Switch — blocks ALL new orders for rest of session.
        Existing positions are NOT automatically exited; use exit_all() for that.
        IRREVERSIBLE for the day — only resets next trading session.
        """
        log.critical("DhanTools: ACTIVATING KILL SWITCH")
        return await self._run(self._dhan.kill_switch, "ACTIVATE")

    async def deactivate_kill_switch(self) -> dict:
        """Deactivate kill switch (only valid at session start, not intraday)."""
        log.warning("DhanTools: deactivating kill switch")
        return await self._run(self._dhan.kill_switch, "DEACTIVATE")

    async def set_pnl_exit(
        self,
        max_profit: Optional[float] = None,
        max_loss: Optional[float] = None,
    ) -> dict:
        """
        Configure DhanHQ P&L-based auto-exit.

        max_profit : auto-exit all positions when day P&L hits this ₹ gain
        max_loss   : auto-exit all positions when day P&L hits this ₹ loss
        """
        log.warning(
            "DhanTools: setting P&L exit — profit=₹%s loss=₹%s", max_profit, max_loss
        )
        return await self._run(
            self._dhan.set_dhan_pnl_exit,
            max_profit=max_profit,
            max_loss=max_loss,
        )

    async def exit_all_positions(self) -> dict:
        """
        Exit all open positions immediately (market order).
        Used during CRISIS regime or daily loss circuit breaker.
        """
        log.critical("DhanTools: EXIT ALL POSITIONS")
        return await self._run(self._dhan.exit_all)

    # ═══════════════════════════════════════════════════════════════════════════
    # D. CONVENIENCE HELPERS
    # ═══════════════════════════════════════════════════════════════════════════

    async def get_nifty_ltp(self) -> float:
        """Return NIFTY 50 index LTP (security_id=13, NSE_IDX)."""
        try:
            result = await self.get_quote({"NSE_IDX": ["13"]})
            data = result.get("data", {})
            nifty = data.get("NSE_IDX", {}).get("13", {})
            return float(nifty.get("last_price", 0.0))
        except Exception as exc:
            log.error("DhanTools: failed to get Nifty LTP: %s", exc)
            return 0.0

    async def get_nifty_ohlc_today(self) -> dict:
        """Return today's OHLC for NIFTY 50."""
        try:
            result = await self.get_ohlc({"NSE_IDX": ["13"]})
            data = result.get("data", {})
            return data.get("NSE_IDX", {}).get("13", {})
        except Exception as exc:
            log.error("DhanTools: failed to get Nifty OHLC: %s", exc)
            return {}

    async def get_advance_decline_ratio(self) -> float:
        """
        Approximate NSE advance/decline ratio using a basket of Nifty 500 stocks.
        Returns advances / declines (> 1.0 = more advancing stocks).

        Note: Accurate AD data is fetched by DataTools.get_advance_decline().
        This is a fast approximation using DhanHQ quote for liquid stocks.
        """
        # Security IDs for 20 liquid NSE large-caps (covers representative breadth)
        SAMPLE_IDS = [
            "1333", "11536", "1348", "14366", "15083",   # HDFC, INFY, ICICI, TCS, RIL
            "1270", "6191", "10999", "16669", "2885",     # AXIS, BAJFIN, KOTAK, MARUTI, SBI
            "7229", "13538", "16350", "11630", "2029",    # TATAMOTORS, TITAN, WIPRO, LT, SUNPHARMA
            "772", "4717", "6386", "9519", "1232",        # ASIANPAINT, BHARTIARTL, HCLTECH, NESTLEIND, ONGC
        ]
        try:
            result = await self.get_quote({"NSE_EQ": SAMPLE_IDS})
            quotes = result.get("data", {}).get("NSE_EQ", {})
            advances = sum(
                1 for q in quotes.values()
                if float(q.get("last_price", 0)) > float(q.get("previous_close_price", 0))
            )
            declines = len(quotes) - advances
            return advances / max(declines, 1)
        except Exception as exc:
            log.error("DhanTools: advance/decline failed: %s", exc)
            return 1.0   # neutral fallback
