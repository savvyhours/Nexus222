"""
NEXUS-II — Technical Indicator Compute Tools

Pure-Python / pandas-ta implementations of every indicator needed by the
TechnicalAnalyst, ScalperAgent, TrendFollowerAgent, MeanReversionAgent, and
PatternAgent.

All functions accept a pandas DataFrame with columns:
    open, high, low, close, volume   (lowercase, float)

and return a dict of indicator values (latest bar unless noted).

Depends on: pandas, pandas-ta
Optional:   numpy (for Z-score / pairs stat arb calculations)

No network I/O — these are pure compute functions, fully synchronous.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

log = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe_last(series: "pd.Series", default: float = 0.0) -> float:
    """Return last non-NaN value of a Series, or default."""
    try:
        val = series.dropna().iloc[-1]
        return float(val)
    except (IndexError, TypeError):
        return default


def ohlcv_to_df(bars: list[dict]) -> pd.DataFrame:
    """
    Convert a list of OHLCV dicts (from DhanHQ or any source) to a DataFrame.

    Input dicts may have keys: open/high/low/close/volume (any case).
    Returns a DataFrame indexed by integer with lowercase column names.
    """
    if not bars:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(bars)
    df.columns = [c.lower() for c in df.columns]
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["close"]).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MOMENTUM INDICATORS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_rsi(df: pd.DataFrame, period: int = 14) -> float:
    """RSI(period) of the close series. Returns latest value."""
    try:
        import pandas_ta as ta
        rsi = ta.rsi(df["close"], length=period)
        return _safe_last(rsi, 50.0)
    except Exception as exc:
        log.debug("compute_rsi failed: %s", exc)
        return 50.0


def compute_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> dict[str, float]:
    """
    MACD indicator.

    Returns
    -------
    dict with keys: macd_line, macd_signal, macd_hist
    """
    try:
        import pandas_ta as ta
        result = ta.macd(df["close"], fast=fast, slow=slow, signal=signal)
        if result is None or result.empty:
            return {"macd_line": 0.0, "macd_signal": 0.0, "macd_hist": 0.0}
        cols = result.columns.tolist()
        return {
            "macd_line":   _safe_last(result[cols[0]]),
            "macd_hist":   _safe_last(result[cols[1]]),
            "macd_signal": _safe_last(result[cols[2]]),
        }
    except Exception as exc:
        log.debug("compute_macd failed: %s", exc)
        return {"macd_line": 0.0, "macd_signal": 0.0, "macd_hist": 0.0}


# ═══════════════════════════════════════════════════════════════════════════════
# TREND INDICATORS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ema(df: pd.DataFrame, period: int) -> float:
    """EMA(period) of close. Returns latest value."""
    try:
        import pandas_ta as ta
        ema = ta.ema(df["close"], length=period)
        return _safe_last(ema)
    except Exception as exc:
        log.debug("compute_ema(%d) failed: %s", period, exc)
        return _safe_last(df["close"])


def compute_ema_ribbon(df: pd.DataFrame) -> dict[str, float]:
    """
    EMA ribbon: 8, 13, 21, 34, 55 period EMAs.

    Returns dict: ema_8, ema_13, ema_21, ema_34, ema_55.
    All aligned upward (ema_8 > ema_13 > ... > ema_55) = strong bull.
    All aligned downward = strong bear.
    """
    periods = [8, 13, 21, 34, 55]
    return {f"ema_{p}": compute_ema(df, p) for p in periods}


def compute_adx(df: pd.DataFrame, period: int = 14) -> dict[str, float]:
    """
    ADX, +DI, -DI.

    Returns dict: adx, plus_di, minus_di.
    ADX > 25 = trend confirmed; < 20 = range-bound.
    """
    try:
        import pandas_ta as ta
        result = ta.adx(df["high"], df["low"], df["close"], length=period)
        if result is None or result.empty:
            return {"adx": 20.0, "plus_di": 0.0, "minus_di": 0.0}
        cols = result.columns.tolist()
        return {
            "adx":      _safe_last(result[cols[0]], 20.0),
            "plus_di":  _safe_last(result[cols[1]]),
            "minus_di": _safe_last(result[cols[2]]),
        }
    except Exception as exc:
        log.debug("compute_adx failed: %s", exc)
        return {"adx": 20.0, "plus_di": 0.0, "minus_di": 0.0}


def compute_ema_trend(df: pd.DataFrame) -> dict[str, float]:
    """
    EMA(20) and EMA(50) for golden-cross / death-cross detection.

    Returns: ema_20, ema_50, crossover_signal (+1 golden, -1 death, 0 neutral)
    """
    ema20 = compute_ema(df, 20)
    ema50 = compute_ema(df, 50)
    if len(df) >= 2:
        prev_ema20 = _safe_last(
            pd.Series(df["close"].ewm(span=20).mean().iloc[:-1])
        )
        prev_ema50 = _safe_last(
            pd.Series(df["close"].ewm(span=50).mean().iloc[:-1])
        )
        if prev_ema20 < prev_ema50 and ema20 > ema50:
            crossover = 1.0    # golden cross
        elif prev_ema20 > prev_ema50 and ema20 < ema50:
            crossover = -1.0   # death cross
        else:
            crossover = 0.0
    else:
        crossover = 0.0
    return {"ema_20": ema20, "ema_50": ema50, "crossover_signal": crossover}


def compute_52w_proximity(df: pd.DataFrame) -> dict[str, float]:
    """
    52-week high/low and proximity to 52-week high.

    Uses the last 252 bars of df.  Returns:
        high_52w, low_52w, pct_from_52w_high (negative = below high)
    """
    window = df.tail(252)
    high = float(window["high"].max()) if not window.empty else 0.0
    low  = float(window["low"].min())  if not window.empty else 0.0
    ltp  = _safe_last(df["close"])
    pct_from_high = (ltp - high) / high if high > 0 else 0.0
    return {"high_52w": high, "low_52w": low, "pct_from_52w_high": round(pct_from_high, 4)}


# ═══════════════════════════════════════════════════════════════════════════════
# VOLATILITY INDICATORS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_atr(df: pd.DataFrame, period: int = 14) -> dict[str, float]:
    """
    ATR(period) and 20-bar ATR average.

    Returns: atr (latest), atr_avg_20.
    """
    try:
        import pandas_ta as ta
        atr_series = ta.atr(df["high"], df["low"], df["close"], length=period)
        atr_val    = _safe_last(atr_series)
        atr_avg20  = float(atr_series.tail(20).mean()) if atr_series is not None else atr_val
        return {"atr": atr_val, "atr_avg_20": atr_avg20}
    except Exception as exc:
        log.debug("compute_atr failed: %s", exc)
        return {"atr": 0.0, "atr_avg_20": 0.0}


def compute_bollinger(
    df: pd.DataFrame, period: int = 20, std: float = 2.0
) -> dict[str, float]:
    """
    Bollinger Bands: upper, mid, lower, width.

    width = (upper - lower) / mid — smaller width = squeeze (volatility contraction).
    """
    try:
        import pandas_ta as ta
        result = ta.bbands(df["close"], length=period, std=std)
        if result is None or result.empty:
            close = _safe_last(df["close"])
            return {"bb_upper": close, "bb_mid": close, "bb_lower": close, "bb_width": 0.0}
        cols = result.columns.tolist()
        lower  = _safe_last(result[cols[0]])
        mid    = _safe_last(result[cols[1]])
        upper  = _safe_last(result[cols[2]])
        width  = (upper - lower) / mid if mid != 0 else 0.0
        return {"bb_lower": lower, "bb_mid": mid, "bb_upper": upper, "bb_width": round(width, 4)}
    except Exception as exc:
        log.debug("compute_bollinger failed: %s", exc)
        close = _safe_last(df["close"])
        return {"bb_upper": close, "bb_mid": close, "bb_lower": close, "bb_width": 0.0}


# ═══════════════════════════════════════════════════════════════════════════════
# VOLUME INDICATORS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_obv(df: pd.DataFrame) -> float:
    """On-Balance Volume (latest value)."""
    try:
        import pandas_ta as ta
        obv = ta.obv(df["close"], df["volume"])
        return _safe_last(obv)
    except Exception as exc:
        log.debug("compute_obv failed: %s", exc)
        return 0.0


def compute_vwap(df: pd.DataFrame) -> float:
    """
    VWAP for the session bars in `df`.

    Standard formula: cumulative(typical_price × volume) / cumulative(volume).
    """
    try:
        tp = (df["high"] + df["low"] + df["close"]) / 3
        cum_tp_vol = (tp * df["volume"]).cumsum()
        cum_vol    = df["volume"].cumsum()
        vwap_series = cum_tp_vol / cum_vol
        return _safe_last(vwap_series)
    except Exception as exc:
        log.debug("compute_vwap failed: %s", exc)
        return _safe_last(df["close"])


def compute_volume_spike(df: pd.DataFrame, lookback: int = 20) -> dict[str, float]:
    """
    Detect volume spikes vs rolling average.

    Returns: volume (latest), volume_avg_20, volume_ratio (current / avg).
    Ratio > 2.0 = significant spike (ScalperAgent threshold).
    """
    current_vol = _safe_last(df["volume"])
    avg_vol     = float(df["volume"].tail(lookback).mean()) if len(df) >= lookback else current_vol
    ratio       = current_vol / avg_vol if avg_vol > 0 else 1.0
    return {
        "volume":        current_vol,
        "volume_avg_20": avg_vol,
        "volume_ratio":  round(ratio, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PIVOT / INTRADAY LEVELS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_cpr(prev_high: float, prev_low: float, prev_close: float) -> dict[str, float]:
    """
    Central Pivot Range from previous day's H/L/C.

    Returns: cpr_pivot (P), cpr_bc (bottom central), cpr_tc (top central),
             s1, s2, r1, r2.
    """
    pivot = (prev_high + prev_low + prev_close) / 3
    bc    = (prev_high + prev_low) / 2
    tc    = pivot - bc + pivot
    r1    = 2 * pivot - prev_low
    s1    = 2 * pivot - prev_high
    r2    = pivot + (prev_high - prev_low)
    s2    = pivot - (prev_high - prev_low)
    return {
        "cpr_pivot": round(pivot, 2),
        "cpr_bc":    round(bc, 2),
        "cpr_tc":    round(tc, 2),
        "r1": round(r1, 2), "r2": round(r2, 2),
        "s1": round(s1, 2), "s2": round(s2, 2),
    }


def compute_orb(df_first_15: pd.DataFrame) -> dict[str, float]:
    """
    Opening Range Breakout levels from the first 15 minutes of data.

    Parameters
    ----------
    df_first_15 : DataFrame with the first 15-minute bars only.

    Returns
    -------
    orb_high, orb_low, orb_range, orb_midpoint.
    """
    if df_first_15.empty:
        return {"orb_high": 0.0, "orb_low": 0.0, "orb_range": 0.0, "orb_midpoint": 0.0}
    h = float(df_first_15["high"].max())
    l = float(df_first_15["low"].min())
    return {
        "orb_high":     round(h, 2),
        "orb_low":      round(l, 2),
        "orb_range":    round(h - l, 2),
        "orb_midpoint": round((h + l) / 2, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STATISTICAL INDICATORS (Mean Reversion / Pairs)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_zscore(series: "pd.Series", window: int = 20) -> float:
    """
    Rolling Z-score of a price series.

    |Z| > 2.0 triggers MeanReversionAgent.
    """
    try:
        mean  = series.rolling(window).mean()
        std   = series.rolling(window).std()
        z     = (series - mean) / std
        return _safe_last(z)
    except Exception:
        return 0.0


def compute_spread_zscore(
    series_a: "pd.Series", series_b: "pd.Series", window: int = 20
) -> float:
    """
    Z-score of the spread between two price series (for pairs / stat-arb).

    Used by MeanReversionAgent for intra-sector pairs trading.
    """
    spread = series_a - series_b
    return compute_zscore(spread, window)


# ═══════════════════════════════════════════════════════════════════════════════
# COMPOSITE: compute_all_indicators
# ═══════════════════════════════════════════════════════════════════════════════

def compute_all_indicators(
    df: pd.DataFrame,
    prev_day: Optional[dict] = None,
    df_first_15: Optional[pd.DataFrame] = None,
) -> dict[str, Any]:
    """
    Compute the full indicator set for TechnicalAnalyst and strategy agents.

    Parameters
    ----------
    df          : OHLCV DataFrame (at least 55 bars recommended for EMA ribbon)
    prev_day    : dict with prev_high, prev_low, prev_close for CPR calculation
    df_first_15 : first-15-min bars for ORB (optional; intraday use only)

    Returns
    -------
    dict with all indicator values (floats), ready for TechnicalAnalyst.analyze().
    """
    if df.empty:
        log.warning("compute_all_indicators: empty DataFrame")
        return {}

    result: dict[str, Any] = {}

    # ── Momentum ──
    result["rsi"] = compute_rsi(df)
    result.update(compute_macd(df))

    # ── EMA ribbon + trend ──
    result.update(compute_ema_ribbon(df))
    result.update(compute_ema_trend(df))
    result.update(compute_52w_proximity(df))

    # ── ADX ──
    result.update(compute_adx(df))

    # ── Volatility ──
    result.update(compute_atr(df))
    result.update(compute_bollinger(df))

    # ── Volume ──
    result["obv"]  = compute_obv(df)
    result["vwap"] = compute_vwap(df)
    result.update(compute_volume_spike(df))

    # ── Price levels ──
    if prev_day:
        result.update(compute_cpr(
            prev_day.get("prev_high", 0.0),
            prev_day.get("prev_low",  0.0),
            prev_day.get("prev_close", 0.0),
        ))

    if df_first_15 is not None:
        result.update(compute_orb(df_first_15))

    # ── Current price / gap ──
    result["ltp"]        = _safe_last(df["close"])
    result["open"]       = float(df["open"].iloc[0]) if not df.empty else 0.0
    result["prev_close"] = prev_day.get("prev_close", 0.0) if prev_day else 0.0
    if result["prev_close"] > 0:
        result["gap_pct"] = round(
            (result["open"] - result["prev_close"]) / result["prev_close"], 4
        )
    else:
        result["gap_pct"] = 0.0

    return result
