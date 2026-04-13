"""
NEXUS-II — StrategyLibrary
22 backtested strategies returning a StrategySignal (direction, strength, name).

All strategies are pure functions: (market_data: dict) -> StrategySignal.
They operate on pre-computed indicator values — no network calls.

Source groupings:
  A. Re.Define intraday strategies      (4)
  B. Candlestick composites             (5)  — highest PF on Nifty 18.5yr backtest
  C. Quantitative / factor strategies   (3)
  D. Options strategies                 (4)
  E. Trend / momentum strategies        (6)

Each strategy is registered in STRATEGY_REGISTRY for iteration by the scorer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from core.signal_engine.candlestick_patterns import ema_ribbon_cross


# ── Signal type ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StrategySignal:
    name:      str
    direction: int     # +1 BUY, -1 SELL, 0 HOLD/no signal
    strength:  float   # 0.0–1.0
    group:     str     # A/B/C/D/E
    reason:    str

    @property
    def score(self) -> float:
        """Signed strength: positive = bullish, negative = bearish."""
        return self.direction * self.strength

    @classmethod
    def hold(cls, name: str, group: str) -> "StrategySignal":
        return cls(name=name, direction=0, strength=0.0, group=group, reason="No signal")


# ── Indicator accessors ───────────────────────────────────────────────────────

def _ind(md: dict, key: str, default: float = 0.0) -> float:
    return float(md.get("indicators", {}).get(key, default))

def _quote(md: dict, key: str, default: float = 0.0) -> float:
    return float(md.get("quote", {}).get(key, default))

def _mac(md: dict, key: str, default: float = 0.0) -> float:
    return float(md.get("macro", {}).get(key, default))

def _opt(md: dict, key: str, default: float = 0.0) -> float:
    return float(md.get("options", {}).get(key, default))

def _fund(md: dict, key: str, default: float = 0.0) -> float:
    return float(md.get("fundamentals", {}).get(key, default))


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP A — Re.Define Intraday Strategies (4)
# ═══════════════════════════════════════════════════════════════════════════════

def gap_and_go(md: dict) -> StrategySignal:
    """Gap-and-Go: opening gap > 0.5% with volume > 2x avg → momentum continuation."""
    name       = "gap_and_go"
    ltp        = _quote(md, "ltp")
    prev_close = _quote(md, "prev_close")
    volume     = _quote(md, "volume_1m")
    avg_vol    = _ind(md, "avg_volume_1m", 1.0)

    if prev_close <= 0 or ltp <= 0:
        return StrategySignal.hold(name, "A")

    gap_pct  = (ltp - prev_close) / prev_close
    spike    = avg_vol > 0 and volume >= avg_vol * 2.0

    if gap_pct >= 0.005 and spike:
        strength = min(1.0, 0.55 + abs(gap_pct) * 5 + 0.10)
        return StrategySignal(name=name, direction=1, strength=round(strength, 3),
                              group="A", reason=f"Gap up {gap_pct:.2%}, vol spike {volume:.0f} vs avg {avg_vol:.0f}")
    if gap_pct <= -0.005 and spike:
        strength = min(1.0, 0.55 + abs(gap_pct) * 5)
        return StrategySignal(name=name, direction=-1, strength=round(strength, 3),
                              group="A", reason=f"Gap down {gap_pct:.2%}, vol spike")
    return StrategySignal.hold(name, "A")


def vwap_reversal(md: dict) -> StrategySignal:
    """VWAP Reversal: price > 1.5 ATR from VWAP → fade back to VWAP."""
    name = "vwap_reversal"
    ltp  = _quote(md, "ltp")
    vwap = _ind(md, "vwap", ltp)
    atr  = _ind(md, "atr", 0.0)

    if vwap <= 0 or atr <= 0:
        return StrategySignal.hold(name, "A")

    deviation = ltp - vwap
    dev_atr   = abs(deviation) / atr

    if dev_atr < 1.5:
        return StrategySignal.hold(name, "A")

    strength  = min(1.0, 0.50 + (dev_atr - 1.5) * 0.15)
    direction = -1 if deviation > 0 else 1
    return StrategySignal(name=name, direction=direction, strength=round(strength, 3), group="A",
                          reason=f"Price {dev_atr:.1f} ATR from VWAP — revert to {vwap:.2f}")


def orb_15min(md: dict) -> StrategySignal:
    """Opening Range Breakout (15-min) above/below first-15-min high/low with vol."""
    name    = "orb_15min"
    ltp     = _quote(md, "ltp")
    orb_h   = _ind(md, "orb_high")
    orb_l   = _ind(md, "orb_low")
    volume  = _quote(md, "volume_1m")
    avg_vol = _ind(md, "avg_volume_1m", 1.0)
    atr     = _ind(md, "atr", 0.0)

    if orb_h <= 0 or orb_l <= 0:
        return StrategySignal.hold(name, "A")

    vol_ok = avg_vol > 0 and volume >= avg_vol * 1.5

    if ltp > orb_h and vol_ok:
        strength = min(1.0, 0.60 + (ltp - orb_h) / max(atr, 1e-9) * 0.1)
        return StrategySignal(name=name, direction=1, strength=round(strength, 3),
                              group="A", reason=f"ORB breakout above {orb_h:.2f} with vol spike")
    if ltp < orb_l and vol_ok:
        strength = min(1.0, 0.60 + (orb_l - ltp) / max(atr, 1e-9) * 0.1)
        return StrategySignal(name=name, direction=-1, strength=round(strength, 3),
                              group="A", reason=f"ORB breakdown below {orb_l:.2f} with vol spike")
    return StrategySignal.hold(name, "A")


def cpr_support_resistance(md: dict) -> StrategySignal:
    """CPR (Central Pivot Range): buy at BC support, sell at TC resistance."""
    name      = "cpr_sr"
    ltp       = _quote(md, "ltp")
    cpr_bc    = _ind(md, "cpr_bc")
    cpr_tc    = _ind(md, "cpr_tc")
    atr       = _ind(md, "atr", 1.0)

    if cpr_bc <= 0:
        return StrategySignal.hold(name, "A")

    if abs(ltp - cpr_bc) <= 0.2 * atr and ltp > cpr_bc * 0.998:
        return StrategySignal(name=name, direction=1, strength=0.60, group="A",
                              reason=f"Bouncing at CPR BC support {cpr_bc:.2f}")
    if abs(ltp - cpr_tc) <= 0.2 * atr and ltp < cpr_tc * 1.002:
        return StrategySignal(name=name, direction=-1, strength=0.58, group="A",
                              reason=f"Rejecting at CPR TC resistance {cpr_tc:.2f}")
    if ltp > cpr_tc * 1.003:
        return StrategySignal(name=name, direction=1, strength=0.62, group="A",
                              reason=f"Cleared CPR ({cpr_tc:.2f}) — bullish breakout")
    if ltp < cpr_bc * 0.997:
        return StrategySignal(name=name, direction=-1, strength=0.60, group="A",
                              reason=f"Broke below CPR ({cpr_bc:.2f}) — bearish")
    return StrategySignal.hold(name, "A")


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP B — Candlestick Composites (5) — backtested on 18.5yr Nifty
# ═══════════════════════════════════════════════════════════════════════════════

def strategy_rsi_divergence_hammer(md: dict) -> StrategySignal:
    """RSI Divergence + Hammer (PF 8.51, WR 73.7%). Highest-PF composite."""
    name    = "rsi_divergence_hammer"
    bars    = md.get("ohlcv", [])
    rsi     = _ind(md, "rsi_14", 50.0)
    rsi_div = bool(md.get("indicators", {}).get("rsi_divergence", False))
    from core.signal_engine.candlestick_patterns import rsi_divergence_hammer as _fn
    r = _fn(bars, rsi, rsi_div)
    if r.detected:
        return StrategySignal(name=name, direction=1, strength=r.strength,
                              group="B", reason="RSI bullish divergence + Hammer, RSI oversold")
    return StrategySignal.hold(name, "B")


def strategy_ema_ribbon_cross(md: dict) -> StrategySignal:
    """EMA Ribbon Cross 8/13/21/34/55 (PF 6.72, WR 77.8%)."""
    name = "ema_ribbon_cross"
    bars = md.get("ohlcv", [])
    ind  = md.get("indicators", {})
    ev   = {p: float(ind.get(f"ema_{p}", 0.0)) for p in [8, 13, 21, 34, 55]}
    r    = ema_ribbon_cross(bars, ev)
    if r.detected:
        direction = 1 if r.bullish else -1
        return StrategySignal(name=name, direction=direction, strength=r.strength,
                              group="B", reason=f"EMA ribbon {'bull' if r.bullish else 'bear'} aligned")
    return StrategySignal.hold(name, "B")


def strategy_three_white_soldiers(md: dict) -> StrategySignal:
    """Three White Soldiers (PF 2.90, WR 68.9%)."""
    name = "three_white_soldiers"
    bars = md.get("ohlcv", [])
    from core.signal_engine.candlestick_patterns import three_white_soldiers as _fn
    r = _fn(bars)
    if r.detected:
        return StrategySignal(name=name, direction=1, strength=r.strength,
                              group="B", reason="Three White Soldiers: 3 consecutive bullish candles")
    return StrategySignal.hold(name, "B")


def strategy_engulfing_volume(md: dict) -> StrategySignal:
    """Engulfing + Volume Spike (PF 2.45, WR 65.2%)."""
    name    = "engulfing_volume_spike"
    bars    = md.get("ohlcv", [])
    volume  = _quote(md, "volume_1m")
    avg_vol = _ind(md, "avg_volume_1m", 1.0)
    ratio   = volume / avg_vol if avg_vol > 0 else 1.0
    from core.signal_engine.candlestick_patterns import engulfing_volume_spike as _fn
    r = _fn(bars, ratio)
    if r.detected:
        direction = 1 if r.bullish else -1
        return StrategySignal(name=name, direction=direction, strength=r.strength,
                              group="B", reason=f"{'Bullish' if r.bullish else 'Bearish'} engulfing + vol {ratio:.1f}x")
    return StrategySignal.hold(name, "B")


def strategy_morning_star_rsi(md: dict) -> StrategySignal:
    """Morning Star + RSI < 30 (PF 2.12, WR 62.1%)."""
    name = "morning_star_rsi"
    bars = md.get("ohlcv", [])
    rsi  = _ind(md, "rsi_14", 50.0)
    from core.signal_engine.candlestick_patterns import morning_star_rsi as _fn
    r = _fn(bars, rsi)
    if r.detected:
        return StrategySignal(name=name, direction=1, strength=r.strength,
                              group="B", reason=f"Morning Star + RSI={rsi:.1f} < 30")
    return StrategySignal.hold(name, "B")


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP C — Quantitative Strategies (3)
# ═══════════════════════════════════════════════════════════════════════════════

def qlib_factor_ranking(md: dict) -> StrategySignal:
    """Qlib Alpha158: top-decile → BUY, bottom-decile → SELL."""
    name     = "qlib_factor_ranking"
    rank_pct = _fund(md, "qlib_rank_percentile", 50.0)
    pred_ret = _fund(md, "qlib_predicted_return_5d", 0.0)

    if rank_pct >= 90 and pred_ret > 0:
        strength = min(1.0, 0.55 + (rank_pct - 90) / 10 * 0.30 + min(0.10, pred_ret * 5))
        return StrategySignal(name=name, direction=1, strength=round(strength, 3),
                              group="C", reason=f"Qlib top decile rank={rank_pct:.0f}th, pred_5d={pred_ret:.2%}")
    if rank_pct <= 10 and pred_ret < 0:
        strength = min(1.0, 0.55 + (10 - rank_pct) / 10 * 0.30 + min(0.10, abs(pred_ret) * 5))
        return StrategySignal(name=name, direction=-1, strength=round(strength, 3),
                              group="C", reason=f"Qlib bottom decile rank={rank_pct:.0f}th, pred_5d={pred_ret:.2%}")
    return StrategySignal.hold(name, "C")


def statistical_arbitrage(md: dict) -> StrategySignal:
    """Pairs stat-arb: Z-score of spread > 2σ → mean-revert."""
    name   = "stat_arb"
    zscore = _ind(md, "pair_zscore", 0.0)
    pair   = md.get("indicators", {}).get("pair_symbol", "pair")

    if abs(zscore) < 2.0:
        return StrategySignal.hold(name, "C")

    direction = -1 if zscore > 0 else 1
    strength  = min(1.0, 0.50 + (abs(zscore) - 2.0) * 0.10)
    side      = "expensive" if zscore > 0 else "cheap"
    return StrategySignal(name=name, direction=direction, strength=round(strength, 3), group="C",
                          reason=f"Stat arb: {side} vs {pair} (Z={zscore:.2f})")


def momentum_factor_20d(md: dict) -> StrategySignal:
    """Momentum factor: 20-day return rank > 85th → BUY, < 15th → SELL."""
    name     = "momentum_20d"
    mom_rank = _fund(md, "momentum_20d_rank", 50.0)
    ret_20d  = _ind(md, "return_20d", 0.0)

    if mom_rank >= 85:
        strength = min(1.0, 0.55 + (mom_rank - 85) / 15 * 0.30)
        return StrategySignal(name=name, direction=1, strength=round(strength, 3),
                              group="C", reason=f"Top momentum rank={mom_rank:.0f}th, 20d_ret={ret_20d:.1%}")
    if mom_rank <= 15:
        strength = min(1.0, 0.55 + (15 - mom_rank) / 15 * 0.30)
        return StrategySignal(name=name, direction=-1, strength=round(strength, 3),
                              group="C", reason=f"Negative momentum rank={mom_rank:.0f}th, 20d_ret={ret_20d:.1%}")
    return StrategySignal.hold(name, "C")


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP D — Options Strategies (4)
# ═══════════════════════════════════════════════════════════════════════════════

def iv_crush(md: dict) -> StrategySignal:
    """IV Crush: sell ATM straddle pre-earnings when IV > 80th pct, DTE <= 7."""
    name     = "iv_crush"
    iv_pct   = _opt(md, "iv_percentile", 50.0)
    dte      = int(_opt(md, "days_to_expiry", 30))
    earnings = bool(md.get("options", {}).get("earnings_imminent", False))

    if iv_pct > 80 and dte <= 7 and earnings:
        strength = min(1.0, 0.60 + (iv_pct - 80) / 20 * 0.25)
        return StrategySignal(name=name, direction=-1, strength=round(strength, 3),
                              group="D", reason=f"IV crush: IV={iv_pct:.0f}th pct, DTE={dte}, earnings near")
    return StrategySignal.hold(name, "D")


def theta_decay(md: dict) -> StrategySignal:
    """Theta Decay: sell OTM covered calls — DTE <= 14, IV >= 40th pct, VIX < 22."""
    name   = "theta_decay"
    iv_pct = _opt(md, "iv_percentile", 50.0)
    dte    = int(_opt(md, "days_to_expiry", 30))
    vix    = _mac(md, "india_vix", 16.0)

    if dte <= 14 and iv_pct >= 40 and vix < 22:
        strength = min(1.0, 0.55 + (14 - dte) / 14 * 0.20)
        return StrategySignal(name=name, direction=-1, strength=round(strength, 3),
                              group="D", reason=f"Theta decay: DTE={dte}, IV={iv_pct:.0f}th, VIX={vix:.1f}")
    return StrategySignal.hold(name, "D")


def vix_hedge(md: dict) -> StrategySignal:
    """VIX Hedge: buy ATM PE when VIX >= 18. Portfolio protection."""
    name    = "vix_hedge_pe"
    vix     = _mac(md, "india_vix", 16.0)
    trigger = 18.0

    if vix >= trigger:
        strength = min(1.0, 0.50 + (vix - trigger) / 15 * 0.35)
        return StrategySignal(name=name, direction=1, strength=round(strength, 3),
                              group="D", reason=f"VIX={vix:.1f} >= {trigger} — buy PE hedge")
    return StrategySignal.hold(name, "D")


def max_pain_positioning(md: dict) -> StrategySignal:
    """Max Pain: DTE <= 2, position toward max pain strike."""
    name     = "max_pain"
    ltp      = _quote(md, "ltp")
    max_pain = _opt(md, "max_pain_strike", 0.0)
    dte      = int(_opt(md, "days_to_expiry", 30))

    if max_pain <= 0 or dte > 2 or ltp <= 0:
        return StrategySignal.hold(name, "D")

    diff_pct = (max_pain - ltp) / ltp
    if abs(diff_pct) < 0.01:
        return StrategySignal.hold(name, "D")

    direction = 1 if diff_pct > 0 else -1
    strength  = min(1.0, 0.55 + abs(diff_pct) * 5)
    return StrategySignal(name=name, direction=direction, strength=round(strength, 3), group="D",
                          reason=f"Max pain={max_pain:.0f} vs LTP={ltp:.0f} ({diff_pct:+.1%}), DTE={dte}")


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP E — Trend / Momentum Strategies (6)
# ═══════════════════════════════════════════════════════════════════════════════

def ema_crossover_20_50(md: dict) -> StrategySignal:
    """EMA(20) vs EMA(50) crossover with ADX > 25 trend filter and MACD confirmation."""
    name     = "ema_20_50_crossover"
    ema20    = _ind(md, "ema_20")
    ema50    = _ind(md, "ema_50")
    adx      = _ind(md, "adx", 0.0)
    macd     = _ind(md, "macd_line", 0.0)
    macd_sig = _ind(md, "macd_signal", 0.0)

    if ema20 <= 0 or ema50 <= 0 or adx < 25:
        return StrategySignal.hold(name, "E")

    if ema20 > ema50 and macd > macd_sig:
        strength = min(1.0, 0.60 + (adx - 25) / 50 * 0.25)
        return StrategySignal(name=name, direction=1, strength=round(strength, 3),
                              group="E", reason=f"EMA20>{ema50:.0f}, ADX={adx:.1f}, MACD bullish")
    if ema20 < ema50 and macd < macd_sig:
        strength = min(1.0, 0.60 + (adx - 25) / 50 * 0.25)
        return StrategySignal(name=name, direction=-1, strength=round(strength, 3),
                              group="E", reason=f"EMA20<EMA50, ADX={adx:.1f}, MACD bearish")
    return StrategySignal.hold(name, "E")


def breakout_52w_high(md: dict) -> StrategySignal:
    """52-week high breakout within 0.5% with ADX > 25 and volume confirmation."""
    name   = "52w_high_breakout"
    ltp    = _quote(md, "ltp")
    h52    = _ind(md, "high_52w")
    adx    = _ind(md, "adx", 0.0)
    volume = _quote(md, "volume_1m")
    avg_v  = _ind(md, "avg_volume_1m", 1.0)

    if h52 <= 0 or adx < 25:
        return StrategySignal.hold(name, "E")

    vol_ok = avg_v > 0 and volume >= avg_v * 1.2
    if ltp >= h52 * 0.995 and vol_ok:
        return StrategySignal(name=name, direction=1, strength=0.75, group="E",
                              reason=f"52w high breakout: LTP={ltp:.2f} ≥ {h52:.2f}, ADX={adx:.1f}")
    return StrategySignal.hold(name, "E")


def macd_histogram_momentum(md: dict) -> StrategySignal:
    """MACD Histogram expanding in same direction for 3 bars — early trend entry."""
    name       = "macd_histogram"
    hist       = _ind(md, "macd_hist", 0.0)
    hist_prev  = _ind(md, "macd_hist_prev", 0.0)
    hist_prev2 = _ind(md, "macd_hist_prev2", 0.0)

    if hist > hist_prev > hist_prev2 > 0:
        strength = min(1.0, 0.55 + abs(hist) / max(abs(hist_prev), 1e-9) * 0.10)
        return StrategySignal(name=name, direction=1, strength=round(strength, 3),
                              group="E", reason=f"MACD hist expanding bullish ({hist:.4f})")
    if hist < hist_prev < hist_prev2 < 0:
        strength = min(1.0, 0.55 + abs(hist) / max(abs(hist_prev), 1e-9) * 0.10)
        return StrategySignal(name=name, direction=-1, strength=round(strength, 3),
                              group="E", reason=f"MACD hist expanding bearish ({hist:.4f})")
    return StrategySignal.hold(name, "E")


def rsi_trend_confirmation(md: dict) -> StrategySignal:
    """RSI in trend zone (>55 bull / <45 bear) with ADX > 20 — secondary confirmation."""
    name = "rsi_trend_zone"
    rsi  = _ind(md, "rsi_14", 50.0)
    adx  = _ind(md, "adx", 0.0)

    if adx < 20:
        return StrategySignal.hold(name, "E")

    if rsi > 55:
        strength = min(1.0, 0.40 + (rsi - 55) / 45 * 0.35)
        return StrategySignal(name=name, direction=1, strength=round(strength, 3),
                              group="E", reason=f"RSI={rsi:.1f} bull zone, ADX={adx:.1f}")
    if rsi < 45:
        strength = min(1.0, 0.40 + (45 - rsi) / 45 * 0.35)
        return StrategySignal(name=name, direction=-1, strength=round(strength, 3),
                              group="E", reason=f"RSI={rsi:.1f} bear zone, ADX={adx:.1f}")
    return StrategySignal.hold(name, "E")


def obv_trend_divergence(md: dict) -> StrategySignal:
    """OBV confirmation/divergence: OBV matching price = continuation; divergence = reversal."""
    name       = "obv_divergence"
    ltp        = _quote(md, "ltp")
    prev_close = _quote(md, "prev_close")
    obv        = _ind(md, "obv", 0.0)
    obv_prev   = _ind(md, "obv_prev", 0.0)

    if prev_close <= 0 or obv_prev == 0:
        return StrategySignal.hold(name, "E")

    price_up = ltp > prev_close
    obv_up   = obv > obv_prev

    if price_up and obv_up:
        return StrategySignal(name=name, direction=1, strength=0.58, group="E",
                              reason="OBV confirming price rise — accumulation")
    if not price_up and not obv_up:
        return StrategySignal(name=name, direction=-1, strength=0.58, group="E",
                              reason="OBV confirming price fall — distribution")
    if price_up and not obv_up:
        return StrategySignal(name=name, direction=-1, strength=0.52, group="E",
                              reason="Bearish OBV divergence: price up, OBV falling")
    # not price_up and obv_up
    return StrategySignal(name=name, direction=1, strength=0.52, group="E",
                          reason="Bullish OBV divergence: price down, OBV rising")


def bollinger_squeeze_breakout(md: dict) -> StrategySignal:
    """BB squeeze (width < 50% of 20-bar avg) followed by directional breakout."""
    name         = "bb_squeeze_breakout"
    ltp          = _quote(md, "ltp")
    bb_upper     = _ind(md, "bb_upper")
    bb_lower     = _ind(md, "bb_lower")
    bb_width     = _ind(md, "bb_width")
    bb_width_avg = _ind(md, "bb_width_avg_20", bb_width * 2 if bb_width else 1.0)
    atr          = _ind(md, "atr", 1.0)

    if bb_upper <= 0 or bb_lower <= 0:
        return StrategySignal.hold(name, "E")

    was_squeezed = bb_width_avg > 0 and bb_width < 0.5 * bb_width_avg
    if not was_squeezed:
        return StrategySignal.hold(name, "E")

    if ltp > bb_upper:
        strength = min(1.0, 0.65 + (ltp - bb_upper) / max(atr, 1e-9) * 0.10)
        return StrategySignal(name=name, direction=1, strength=round(strength, 3),
                              group="E", reason=f"BB squeeze breakout UP above {bb_upper:.2f}")
    if ltp < bb_lower:
        strength = min(1.0, 0.65 + (bb_lower - ltp) / max(atr, 1e-9) * 0.10)
        return StrategySignal(name=name, direction=-1, strength=round(strength, 3),
                              group="E", reason=f"BB squeeze breakout DOWN below {bb_lower:.2f}")

    return StrategySignal.hold(name, "E")


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY REGISTRY (22 strategies)
# ═══════════════════════════════════════════════════════════════════════════════

STRATEGY_REGISTRY: list[Callable[[dict], StrategySignal]] = [
    # Group A — Intraday (4)
    gap_and_go, vwap_reversal, orb_15min, cpr_support_resistance,
    # Group B — Candlestick composites (5)
    strategy_rsi_divergence_hammer, strategy_ema_ribbon_cross,
    strategy_three_white_soldiers, strategy_engulfing_volume, strategy_morning_star_rsi,
    # Group C — Quantitative (3)
    qlib_factor_ranking, statistical_arbitrage, momentum_factor_20d,
    # Group D — Options (4)
    iv_crush, theta_decay, vix_hedge, max_pain_positioning,
    # Group E — Trend / momentum (6)
    ema_crossover_20_50, breakout_52w_high, macd_histogram_momentum,
    rsi_trend_confirmation, obv_trend_divergence, bollinger_squeeze_breakout,
]

assert len(STRATEGY_REGISTRY) == 22, f"Expected 22 strategies, got {len(STRATEGY_REGISTRY)}"


def run_all(market_data: dict) -> list[StrategySignal]:
    """Execute all 22 strategies and return every result (including HOLD)."""
    return [fn(market_data) for fn in STRATEGY_REGISTRY]


def active_signals(market_data: dict) -> list[StrategySignal]:
    """Return only non-HOLD signals."""
    return [s for s in run_all(market_data) if s.direction != 0]


def net_score(market_data: dict) -> float:
    """
    Strength-weighted net score across all 22 strategies.
    Returns value in [-1.0, +1.0]; used as the 'strategy_library' sub-component.
    """
    signals = active_signals(market_data)
    if not signals:
        return 0.0
    total = sum(s.strength for s in signals)
    if total == 0:
        return 0.0
    return round(sum(s.score * s.strength for s in signals) / total, 4)
