"""
NEXUS-II — External Data Tools

Async fetchers for data that does NOT come from DhanHQ directly:
  - India VIX        (NSE official endpoint)
  - FII / DII flows  (NSE official endpoint)
  - Advance/Decline  (NSE breadth data)
  - Sector momentum  (NSE sector index prices via DhanHQ)
  - Screener.in      (fundamental data for stocks)
  - News headlines   (Yahoo Finance RSS + Finnhub)
  - Macro data       (USD/INR, Brent crude via AlphaVantage)
  - FRED macro       (Fed Funds Rate, CPI, US 10Y yield, DXY via FRED API)
  - Finnhub data     (company news, earnings calendar, economic calendar)
  - RBI calendar     (static + scrape fallback)

All methods degrade gracefully — they return safe defaults and log errors
rather than raising, so a single feed failure does not abort calibration.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timedelta
from typing import Any, Optional
from zoneinfo import ZoneInfo

import httpx

log = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")

# ── Endpoint constants ─────────────────────────────────────────────────────────

_NSE_VIX_URL        = "https://www.nseindia.com/api/allIndices"
_NSE_FII_DII_URL    = "https://www.nseindia.com/api/fiidiiTradeReact"
_NSE_ADVANCES_URL   = "https://www.nseindia.com/api/equity-stockIndices?index=SECURITIES%20IN%20F%26O"
_NSE_SECTOR_INDICES = {
    "IT":      "NIFTY IT",
    "BANKING": "NIFTY BANK",
    "PHARMA":  "NIFTY PHARMA",
    "AUTO":    "NIFTY AUTO",
    "FMCG":    "NIFTY FMCG",
    "REALTY":  "NIFTY REALTY",
    "METAL":   "NIFTY METAL",
    "ENERGY":  "NIFTY ENERGY",
}
_NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.nseindia.com",
}
_ALPHA_VANTAGE_BASE = "https://www.alphavantage.co/query"
_FRED_BASE          = "https://api.stlouisfed.org/fred/series/observations"
_FINNHUB_BASE       = "https://finnhub.io/api/v1"

# ── HTTP client factory ────────────────────────────────────────────────────────

def _make_client(timeout: float = 10.0) -> httpx.AsyncClient:
    return httpx.AsyncClient(headers=_NSE_HEADERS, timeout=timeout, follow_redirects=True)


class DataTools:
    """
    Async external data fetchers.

    Parameters
    ----------
    alpha_vantage_key : API key for AlphaVantage (FOREX, commodities)
    nse_session_cookie: optional — NSE session cookie for advanced endpoints.
                        If not provided, VIX is fetched via public allIndices endpoint.
    fred_api_key      : FRED API key (fred.stlouisfed.org) for US macro data:
                        Fed Funds Rate, CPI, US 10Y Treasury yield, DXY.
    finnhub_api_key   : Finnhub API key (finnhub.io) for real-time company news,
                        earnings calendar, economic calendar, and market sentiment.
    """

    def __init__(
        self,
        alpha_vantage_key: str = "",
        nse_session_cookie: str = "",
        fred_api_key: str = "",
        finnhub_api_key: str = "",
    ) -> None:
        self._av_key       = alpha_vantage_key
        self._nse_cookie   = nse_session_cookie
        self._fred_key     = fred_api_key
        self._finnhub_key  = finnhub_api_key

    # ── India VIX ─────────────────────────────────────────────────────────────

    async def get_india_vix(self) -> float:
        """
        Fetch current India VIX from NSE's allIndices endpoint.

        Returns 16.0 (neutral default) on failure.
        """
        try:
            async with _make_client() as client:
                r = await client.get(_NSE_VIX_URL)
                r.raise_for_status()
                indices = r.json().get("data", [])
                for idx in indices:
                    if idx.get("index") == "INDIA VIX":
                        return float(idx.get("last", 16.0))
            log.warning("DataTools: India VIX not found in allIndices response")
        except Exception as exc:
            log.error("DataTools: get_india_vix failed: %s", exc)
        return 16.0

    # ── FII / DII flows ────────────────────────────────────────────────────────

    async def get_fii_dii_flows(self, days: int = 5) -> dict:
        """
        Fetch FII and DII net equity flows from NSE.

        Returns
        -------
        dict with keys:
            fii_net_today_cr    : float — today's net FII flow (₹ crore)
            dii_net_today_cr    : float — today's net DII flow (₹ crore)
            fii_net_3d_cr       : float — 3-day cumulative FII flow
            fii_net_5d_cr       : float — 5-day cumulative FII flow
            fii_consecutive_days: int   — consecutive net buying (+) or selling (-)
            history             : list of daily dicts (date, fii_net, dii_net)
        """
        default = {
            "fii_net_today_cr": 0.0,
            "dii_net_today_cr": 0.0,
            "fii_net_3d_cr": 0.0,
            "fii_net_5d_cr": 0.0,
            "fii_consecutive_days": 0,
            "history": [],
        }
        try:
            async with _make_client() as client:
                r = await client.get(_NSE_FII_DII_URL)
                r.raise_for_status()
                raw = r.json()

            records = raw if isinstance(raw, list) else raw.get("data", [])
            history = []
            for rec in records[:days]:
                fii_net = float(rec.get("fiiNetPurchases", 0) or 0)
                dii_net = float(rec.get("diiNetPurchases", 0) or 0)
                history.append({
                    "date": rec.get("date", ""),
                    "fii_net": fii_net,
                    "dii_net": dii_net,
                })

            if not history:
                return default

            today = history[0]
            fii_3d = sum(d["fii_net"] for d in history[:3])
            fii_5d = sum(d["fii_net"] for d in history[:5])

            # Count consecutive buying or selling days
            consecutive = 0
            direction = 1 if today["fii_net"] >= 0 else -1
            for d in history:
                if (d["fii_net"] >= 0) == (direction == 1):
                    consecutive += 1
                else:
                    break
            consecutive *= direction

            return {
                "fii_net_today_cr": today["fii_net"],
                "dii_net_today_cr": today["dii_net"],
                "fii_net_3d_cr": fii_3d,
                "fii_net_5d_cr": fii_5d,
                "fii_consecutive_days": consecutive,
                "history": history,
            }

        except Exception as exc:
            log.error("DataTools: get_fii_dii_flows failed: %s", exc)
            return default

    # ── Advance / Decline ratio ────────────────────────────────────────────────

    async def get_advance_decline_ratio(self) -> float:
        """
        NSE advance/decline ratio from FnO securities list.

        Returns 1.0 (neutral) on failure.
        """
        try:
            async with _make_client() as client:
                r = await client.get(_NSE_ADVANCES_URL)
                r.raise_for_status()
                data = r.json().get("data", [])

            advances = sum(
                1 for s in data
                if float(s.get("change", 0) or 0) > 0
            )
            declines = sum(
                1 for s in data
                if float(s.get("change", 0) or 0) < 0
            )
            return advances / max(declines, 1)

        except Exception as exc:
            log.error("DataTools: get_advance_decline_ratio failed: %s", exc)
            return 1.0

    # ── Sector momentum ────────────────────────────────────────────────────────

    async def get_sector_momentum(self) -> dict[str, float]:
        """
        5-day return % for major NSE sector indices.

        Fetches allIndices and computes (last - prev5d) / prev5d.
        Returns dict: {"IT": 0.021, "BANKING": -0.015, ...}
        Falls back to zeros on failure.
        """
        try:
            async with _make_client() as client:
                r = await client.get(_NSE_VIX_URL)   # allIndices has all sector indices
                r.raise_for_status()
                indices = r.json().get("data", [])

            result: dict[str, float] = {}
            index_map = {v: k for k, v in _NSE_SECTOR_INDICES.items()}

            for idx in indices:
                sector_key = index_map.get(idx.get("index", ""))
                if sector_key:
                    last = float(idx.get("last", 0) or 0)
                    pct_change = float(idx.get("pChange", 0) or 0) / 100
                    result[sector_key] = round(pct_change, 5)

            # fill any missing sectors with 0
            for k in _NSE_SECTOR_INDICES:
                result.setdefault(k, 0.0)

            return result

        except Exception as exc:
            log.error("DataTools: get_sector_momentum failed: %s", exc)
            return {k: 0.0 for k in _NSE_SECTOR_INDICES}

    # ── Nifty change % ────────────────────────────────────────────────────────

    async def get_nifty_change_pct(self) -> float:
        """
        Current Nifty 50 intraday % change from allIndices.

        Returns 0.0 on failure.
        """
        try:
            async with _make_client() as client:
                r = await client.get(_NSE_VIX_URL)
                r.raise_for_status()
                for idx in r.json().get("data", []):
                    if idx.get("index") == "NIFTY 50":
                        return float(idx.get("pChange", 0) or 0) / 100
        except Exception as exc:
            log.error("DataTools: get_nifty_change_pct failed: %s", exc)
        return 0.0

    # ── Macro data (AlphaVantage) ─────────────────────────────────────────────

    async def get_usd_inr(self) -> float:
        """Current USD/INR spot rate via AlphaVantage FX endpoint."""
        if not self._av_key:
            log.debug("DataTools: no AlphaVantage key — USD/INR skipped")
            return 84.0   # safe fallback
        try:
            params = {
                "function": "CURRENCY_EXCHANGE_RATE",
                "from_currency": "USD",
                "to_currency": "INR",
                "apikey": self._av_key,
            }
            async with _make_client() as client:
                r = await client.get(_ALPHA_VANTAGE_BASE, params=params)
                r.raise_for_status()
                data = r.json().get("Realtime Currency Exchange Rate", {})
                return float(data.get("5. Exchange Rate", 84.0))
        except Exception as exc:
            log.error("DataTools: get_usd_inr failed: %s", exc)
            return 84.0

    async def get_brent_crude(self) -> float:
        """Current Brent crude price (USD/barrel) via AlphaVantage commodity endpoint."""
        if not self._av_key:
            return 78.0   # safe fallback
        try:
            params = {
                "function": "BRENT",
                "interval": "daily",
                "apikey": self._av_key,
            }
            async with _make_client() as client:
                r = await client.get(_ALPHA_VANTAGE_BASE, params=params)
                r.raise_for_status()
                rows = r.json().get("data", [])
                if rows:
                    return float(rows[0].get("value", 78.0))
        except Exception as exc:
            log.error("DataTools: get_brent_crude failed: %s", exc)
        return 78.0

    # ── Screener.in fundamentals ──────────────────────────────────────────────

    async def get_screener_fundamentals(self, symbol: str) -> dict:
        """
        Fetch fundamental data for a stock from Screener.in.

        Returns a dict with pe_ratio, eps_growth_yoy, roe, debt_equity,
        promoter_holding, market_cap_cr.  Returns empty dict on failure.

        Note: Screener.in does not have an official API; this uses the
        public JSON endpoint. Rate-limit to < 5 requests/sec.
        """
        url = f"https://www.screener.in/api/company/{symbol}/?format=json"
        try:
            async with _make_client(timeout=15.0) as client:
                r = await client.get(url)
                if r.status_code == 404:
                    log.warning("DataTools: Screener.in 404 for %s", symbol)
                    return {}
                r.raise_for_status()
                data = r.json()

            # Extract key ratios from the ratios list
            ratios = {
                item.get("name"): item.get("values", [{}])[-1].get("value")
                for item in data.get("ratios", [])
                if item.get("values")
            }

            def _pct(key: str, default: float = 0.0) -> float:
                v = ratios.get(key, default)
                try:
                    return float(str(v).replace("%", "").replace(",", "")) / 100
                except (TypeError, ValueError):
                    return default

            def _num(key: str, default: float = 0.0) -> float:
                v = ratios.get(key, default)
                try:
                    return float(str(v).replace(",", ""))
                except (TypeError, ValueError):
                    return default

            return {
                "pe_ratio":         _num("P/E"),
                "eps_growth_yoy":   _pct("EPS Growth"),
                "roe":              _pct("Return on equity"),
                "debt_equity":      _num("Debt to equity"),
                "promoter_holding": _pct("Promoter holding"),
                "dividend_yield":   _pct("Dividend yield"),
                "market_cap_cr":    _num("Market Cap"),
            }

        except Exception as exc:
            log.error("DataTools: get_screener_fundamentals(%s) failed: %s", symbol, exc)
            return {}

    # ── News headlines ─────────────────────────────────────────────────────────

    async def get_news_headlines(self, symbol: str, count: int = 20) -> list[str]:
        """
        Fetch recent news headlines for a stock from Google Finance RSS.

        Returns list of headline strings (up to `count` items).
        Returns empty list on failure (sentiment analyst handles gracefully).
        """
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}.NS&region=IN&lang=en-IN"
        try:
            async with _make_client(timeout=8.0) as client:
                r = await client.get(url)
                r.raise_for_status()
                text = r.text

            # Simple RSS title extraction without external XML parser
            import re
            titles = re.findall(r"<title><!\[CDATA\[(.*?)\]\]></title>", text)
            if not titles:
                titles = re.findall(r"<title>(.*?)</title>", text)
            return [t.strip() for t in titles if t.strip()][:count]

        except Exception as exc:
            log.debug("DataTools: get_news_headlines(%s) failed: %s", symbol, exc)
            return []

    # ── RBI calendar (static + live fallback) ─────────────────────────────────

    async def get_rbi_events(self, lookahead_days: int = 7) -> list[str]:
        """
        Return a list of upcoming RBI / major macro events within `lookahead_days`.

        Uses a static rolling schedule for known policy dates.
        In production, augment with a scraped economic calendar API.
        """
        today = date.today()
        cutoff = today + timedelta(days=lookahead_days)

        # 2026 RBI MPC meeting schedule (update annually)
        RBI_MPC_2026 = [
            date(2026, 2, 7),
            date(2026, 4, 9),
            date(2026, 6, 4),
            date(2026, 8, 6),
            date(2026, 10, 8),
            date(2026, 12, 3),
        ]

        events = []
        for d in RBI_MPC_2026:
            if today <= d <= cutoff:
                events.append(f"RBI MPC decision: {d.strftime('%d %b %Y')}")

        # Budget is typically Feb 1
        budget_date = date(today.year, 2, 1)
        if today <= budget_date <= cutoff:
            events.append(f"Union Budget: {budget_date.strftime('%d %b %Y')}")

        return events

    # ── FRED macro data (US Federal Reserve Economic Data) ───────────────────

    async def _fred_latest(self, series_id: str, default: float) -> float:
        """
        Fetch the latest observation for a FRED series.

        Parameters
        ----------
        series_id : FRED series identifier (e.g. "FEDFUNDS", "CPIAUCSL")
        default   : value returned when key is missing or request fails

        Returns
        -------
        float — most recent observation value
        """
        if not self._fred_key:
            log.debug("DataTools: no FRED key — %s skipped", series_id)
            return default
        try:
            params = {
                "series_id":     series_id,
                "api_key":       self._fred_key,
                "file_type":     "json",
                "sort_order":    "desc",
                "limit":         "1",
            }
            async with _make_client(timeout=10.0) as client:
                r = await client.get(_FRED_BASE, params=params)
                r.raise_for_status()
                obs = r.json().get("observations", [])
                if obs:
                    val = obs[0].get("value", ".")
                    if val != ".":
                        return float(val)
        except Exception as exc:
            log.error("DataTools: FRED %s failed: %s", series_id, exc)
        return default

    async def get_fed_funds_rate(self) -> float:
        """
        Current US Federal Funds Rate (%) from FRED series FEDFUNDS.

        Relevant for Indian markets: Fed rate changes drive DXY and FII flows.
        Returns 5.25 (approximate neutral) on failure.
        """
        return await self._fred_latest("FEDFUNDS", default=5.25)

    async def get_us_cpi_yoy(self) -> float:
        """
        US CPI year-over-year inflation rate (%) from FRED series CPIAUCSL.

        High US inflation → Fed stays hawkish → stronger USD → FII outflows from India.
        Returns 3.0 (approximate neutral) on failure.
        """
        # FRED gives monthly CPI index; compute YoY from last 13 observations
        if not self._fred_key:
            log.debug("DataTools: no FRED key — US CPI skipped")
            return 3.0
        try:
            params = {
                "series_id":  "CPIAUCSL",
                "api_key":    self._fred_key,
                "file_type":  "json",
                "sort_order": "desc",
                "limit":      "13",
            }
            async with _make_client(timeout=10.0) as client:
                r = await client.get(_FRED_BASE, params=params)
                r.raise_for_status()
                obs = r.json().get("observations", [])
                valid = [o for o in obs if o.get("value", ".") != "."]
                if len(valid) >= 13:
                    latest = float(valid[0]["value"])
                    year_ago = float(valid[12]["value"])
                    return round((latest - year_ago) / year_ago * 100, 2)
        except Exception as exc:
            log.error("DataTools: get_us_cpi_yoy failed: %s", exc)
        return 3.0

    async def get_us_10y_yield(self) -> float:
        """
        US 10-Year Treasury yield (%) from FRED series DGS10.

        Rising US yields attract capital away from EMs including India.
        Returns 4.5 (approximate neutral) on failure.
        """
        return await self._fred_latest("DGS10", default=4.5)

    async def get_dxy(self) -> float:
        """
        US Dollar Index (DXY) from FRED series DTWEXBGS (trade-weighted USD).

        Stronger DXY → weaker INR → inflationary pressure + FII outflows.
        Returns 104.0 (approximate neutral) on failure.
        """
        return await self._fred_latest("DTWEXBGS", default=104.0)

    async def get_us_macro_snapshot(self) -> dict:
        """
        Fetch all US macro indicators in a single concurrent call.

        Returns
        -------
        dict with keys:
            fed_funds_rate : float — current Fed Funds Rate (%)
            us_cpi_yoy     : float — US CPI year-over-year inflation (%)
            us_10y_yield   : float — US 10-Year Treasury yield (%)
            dxy            : float — US Dollar Index (trade-weighted)
        """
        fed, cpi, yield_10y, dxy = await asyncio.gather(
            self.get_fed_funds_rate(),
            self.get_us_cpi_yoy(),
            self.get_us_10y_yield(),
            self.get_dxy(),
        )
        return {
            "fed_funds_rate": fed,
            "us_cpi_yoy":     cpi,
            "us_10y_yield":   yield_10y,
            "dxy":            dxy,
        }

    # ── Finnhub data ──────────────────────────────────────────────────────────

    async def get_company_news_finnhub(
        self, symbol: str, count: int = 20
    ) -> list[str]:
        """
        Fetch recent company news headlines from Finnhub.

        Finnhub provides higher-quality, better-curated news than Yahoo RSS.
        Falls back silently to empty list when Finnhub key is absent.

        Parameters
        ----------
        symbol : NSE symbol (e.g. "RELIANCE"). Finnhub uses US tickers by default;
                 for Indian stocks this tries the symbol directly as Finnhub
                 supports BSE/NSE symbols via their exchange mapping.
        count  : maximum number of headlines to return

        Returns
        -------
        list of headline strings
        """
        if not self._finnhub_key:
            log.debug("DataTools: no Finnhub key — company news skipped for %s", symbol)
            return []
        try:
            from datetime import date, timedelta
            today = date.today()
            from_date = (today - timedelta(days=7)).isoformat()
            to_date   = today.isoformat()

            params = {
                "symbol": symbol,
                "from":   from_date,
                "to":     to_date,
                "token":  self._finnhub_key,
            }
            async with _make_client(timeout=10.0) as client:
                r = await client.get(f"{_FINNHUB_BASE}/company-news", params=params)
                r.raise_for_status()
                news = r.json()

            return [
                item.get("headline", "").strip()
                for item in news
                if item.get("headline")
            ][:count]

        except Exception as exc:
            log.debug("DataTools: get_company_news_finnhub(%s) failed: %s", symbol, exc)
            return []

    async def get_earnings_calendar(self, symbol: str) -> list[dict]:
        """
        Fetch upcoming earnings dates for a symbol from Finnhub.

        Returns a list of dicts with keys: date, epsEstimate, revenueEstimate.
        Returns empty list on failure or missing key.
        """
        if not self._finnhub_key:
            return []
        try:
            from datetime import date, timedelta
            today = date.today()
            params = {
                "symbol": symbol,
                "from":   today.isoformat(),
                "to":     (today + timedelta(days=90)).isoformat(),
                "token":  self._finnhub_key,
            }
            async with _make_client(timeout=10.0) as client:
                r = await client.get(f"{_FINNHUB_BASE}/calendar/earnings", params=params)
                r.raise_for_status()
                earnings = r.json().get("earningsCalendar", [])

            return [
                {
                    "date":            e.get("date", ""),
                    "eps_estimate":    e.get("epsEstimate"),
                    "revenue_estimate": e.get("revenueEstimate"),
                }
                for e in earnings
                if e.get("symbol") == symbol
            ]

        except Exception as exc:
            log.debug("DataTools: get_earnings_calendar(%s) failed: %s", symbol, exc)
            return []

    async def get_economic_calendar(self) -> list[dict]:
        """
        Fetch upcoming global economic events from Finnhub.

        Events include US CPI, Fed meetings, NFP, ECB decisions —
        all macro events that drive FII flows into/out of India.

        Returns a list of dicts with keys: time, event, country, impact, estimate.
        Returns empty list on failure.
        """
        if not self._finnhub_key:
            return []
        try:
            params = {"token": self._finnhub_key}
            async with _make_client(timeout=10.0) as client:
                r = await client.get(f"{_FINNHUB_BASE}/calendar/economic", params=params)
                r.raise_for_status()
                events = r.json().get("economicCalendar", [])

            return [
                {
                    "time":     e.get("time", ""),
                    "event":    e.get("event", ""),
                    "country":  e.get("country", ""),
                    "impact":   e.get("impact", ""),
                    "estimate": e.get("estimate"),
                    "actual":   e.get("actual"),
                }
                for e in events
                # Focus on high-impact US + IN events
                if e.get("country") in ("US", "IN", "EU")
                and e.get("impact") in ("high", "medium")
            ]

        except Exception as exc:
            log.debug("DataTools: get_economic_calendar failed: %s", exc)
            return []

    async def get_market_sentiment_finnhub(self) -> dict:
        """
        Fetch bullish/bearish sentiment percentages from Finnhub social sentiment
        for NIFTY (represented via the India ETF "INDA" as a proxy).

        Returns dict with keys: bullish_pct, bearish_pct (floats 0–1).
        Falls back to neutral 0.5/0.5 on failure.
        """
        if not self._finnhub_key:
            return {"bullish_pct": 0.5, "bearish_pct": 0.5}
        try:
            params = {
                "symbol": "INDA",   # iShares MSCI India ETF — best liquid proxy
                "token":  self._finnhub_key,
            }
            async with _make_client(timeout=10.0) as client:
                r = await client.get(f"{_FINNHUB_BASE}/stock/social-sentiment", params=params)
                r.raise_for_status()
                data = r.json()

            reddit  = data.get("reddit", [])
            twitter = data.get("twitter", [])
            combined = reddit + twitter

            if not combined:
                return {"bullish_pct": 0.5, "bearish_pct": 0.5}

            bullish = sum(1 for s in combined if s.get("score", 0) > 0)
            total   = len(combined)
            bullish_pct = round(bullish / total, 3) if total > 0 else 0.5

            return {
                "bullish_pct": bullish_pct,
                "bearish_pct": round(1.0 - bullish_pct, 3),
            }

        except Exception as exc:
            log.debug("DataTools: get_market_sentiment_finnhub failed: %s", exc)
            return {"bullish_pct": 0.5, "bearish_pct": 0.5}

    # ── Average IV percentile ─────────────────────────────────────────────────

    async def get_avg_iv_percentile(
        self, dhan_tools: Any, under_id: str = "13"
    ) -> float:
        """
        Compute the average implied volatility percentile across ATM options
        for NIFTY (or given underlying).

        Requires a DhanTools instance to fetch the options chain.
        Returns 50.0 (neutral) on failure.
        """
        try:
            # Get nearest expiry
            expiry_data = await dhan_tools.get_expiry_list(under_id, "NSE_FNO")
            expiries = expiry_data.get("data", [])
            if not expiries:
                return 50.0
            nearest_expiry = expiries[0]

            chain = await dhan_tools.get_options_chain(under_id, "NSE_FNO", nearest_expiry)
            strikes = chain.get("data", {}).get("oc", {})

            ivs = []
            for strike_data in strikes.values():
                ce_iv = strike_data.get("ce", {}).get("impliedVolatility", 0) or 0
                pe_iv = strike_data.get("pe", {}).get("impliedVolatility", 0) or 0
                if ce_iv > 0:
                    ivs.append(float(ce_iv))
                if pe_iv > 0:
                    ivs.append(float(pe_iv))

            if not ivs:
                return 50.0

            avg_iv = sum(ivs) / len(ivs)
            # Rough percentile: map 8–30% IV range to 0–100 percentile
            percentile = min(100.0, max(0.0, (avg_iv - 8.0) / (30.0 - 8.0) * 100))
            return round(percentile, 1)

        except Exception as exc:
            log.error("DataTools: get_avg_iv_percentile failed: %s", exc)
            return 50.0
