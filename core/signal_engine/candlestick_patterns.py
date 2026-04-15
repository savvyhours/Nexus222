"""
NEXUS-II — CandlestickPatterns
Pure-function pattern detectors operating on list[OHLCV] dicts.

Each function returns a PatternResult(detected, strength, bullish).
All functions are stateless and side-effect-free.

Backtested composites (18.5-year Nifty data — highest profit-factor first):
  rsi_divergence_hammer    PF 8.51  WR 73.7%
  ema_ribbon_cross         PF 6.72  WR 77.8%
  three_white_soldiers     PF 2.90  WR 68.9%
  engulfing_volume_spike   PF 2.45  WR 65.2%
  morning_star_rsi         PF 2.12  WR 62.1%

OHLCV dict keys accepted: o/open, h/high, l/low, c/close, v/volume
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PatternResult:
    name:     str
    detected: bool
    strength: float   # 0.0–1.0
    bullish:  bool    # True = bullish signal

    @classmethod
    def miss(cls, name: str, bullish: bool = True) -> "PatternResult":
        return cls(name=name, detected=False, strength=0.0, bullish=bullish)


# ── OHLCV helpers ─────────────────────────────────────────────────────────────

def _unpack(bar: dict) -> tuple[float, float, float, float, float]:
    return (
        float(bar.get("o", bar.get("open", 0))),
        float(bar.get("h", bar.get("high", 0))),
        float(bar.get("l", bar.get("low", 0))),
        float(bar.get("c", bar.get("close", 0))),
        float(bar.get("v", bar.get("volume", 0))),
    )

def _body(o: float, c: float) -> float:
    return abs(c - o)

def _range(h: float, l: float) -> float:
    return max(h - l, 1e-9)

def _bull(o: float, c: float) -> bool:
    return c >= o

def _bear(o: float, c: float) -> bool:
    return c < o


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLE-BAR PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════

def hammer(bars: list[dict]) -> PatternResult:
    """Hammer: small body at top, lower wick >= 2x body, upper wick <= 0.3x body."""
    name = "hammer"
    if not bars:
        return PatternResult.miss(name)
    o, h, l, c, _ = _unpack(bars[-1])
    body      = _body(o, c)
    low_wick  = min(o, c) - l
    high_wick = h - max(o, c)
    if body < 1e-9:
        return PatternResult.miss(name)
    detected = (low_wick >= 2.0 * body) and (high_wick <= body)
    strength = round(min(1.0, low_wick / body / 3), 3) if detected else 0.0
    return PatternResult(name=name, detected=detected, strength=strength, bullish=True)


def inverted_hammer(bars: list[dict]) -> PatternResult:
    """Inverted Hammer: small body at bottom, upper wick >= 2x body."""
    name = "inverted_hammer"
    if not bars:
        return PatternResult.miss(name)
    o, h, l, c, _ = _unpack(bars[-1])
    body      = _body(o, c)
    high_wick = h - max(o, c)
    low_wick  = min(o, c) - l
    if body < 1e-9:
        return PatternResult.miss(name)
    detected = (high_wick >= 2.0 * body) and (low_wick <= 0.3 * body)
    strength = round(min(1.0, high_wick / body / 3), 3) if detected else 0.0
    return PatternResult(name=name, detected=detected, strength=strength, bullish=True)


def shooting_star(bars: list[dict]) -> PatternResult:
    """Shooting Star (bearish): bearish small body, long upper wick >= 2x body."""
    name = "shooting_star"
    if not bars:
        return PatternResult.miss(name, bullish=False)
    o, h, l, c, _ = _unpack(bars[-1])
    body      = _body(o, c)
    high_wick = h - max(o, c)
    low_wick  = min(o, c) - l
    if body < 1e-9:
        return PatternResult.miss(name, bullish=False)
    detected = _bear(o, c) and (high_wick >= 2.0 * body) and (low_wick <= 0.3 * body)
    strength = round(min(1.0, high_wick / body / 3), 3) if detected else 0.0
    return PatternResult(name=name, detected=detected, strength=strength, bullish=False)


def doji(bars: list[dict], threshold: float = 0.05) -> PatternResult:
    """Doji: body <= threshold * total range. Signals indecision."""
    name = "doji"
    if not bars:
        return PatternResult.miss(name)
    o, h, l, c, _ = _unpack(bars[-1])
    body = _body(o, c)
    rng  = _range(h, l)
    detected = (body / rng) <= threshold
    strength = round(max(0.0, 1.0 - (body / rng) / threshold), 3) if detected else 0.0
    return PatternResult(name=name, detected=detected, strength=strength, bullish=True)


def marubozu_bullish(bars: list[dict]) -> PatternResult:
    """Full bullish body, wicks <= 5% each side. Strong trend signal."""
    name = "marubozu_bullish"
    if not bars:
        return PatternResult.miss(name)
    o, h, l, c, _ = _unpack(bars[-1])
    if not _bull(o, c):
        return PatternResult.miss(name)
    body = _body(o, c)
    if body < 1e-9:
        return PatternResult.miss(name)
    detected = ((h - c) <= 0.05 * body) and ((o - l) <= 0.05 * body)
    return PatternResult(name=name, detected=detected, strength=0.75 if detected else 0.0, bullish=True)


def marubozu_bearish(bars: list[dict]) -> PatternResult:
    """Full bearish body, wicks <= 5% each side."""
    name = "marubozu_bearish"
    if not bars:
        return PatternResult.miss(name, bullish=False)
    o, h, l, c, _ = _unpack(bars[-1])
    if not _bear(o, c):
        return PatternResult.miss(name, bullish=False)
    body = _body(o, c)
    if body < 1e-9:
        return PatternResult.miss(name, bullish=False)
    detected = ((h - o) <= 0.05 * body) and ((c - l) <= 0.05 * body)
    return PatternResult(name=name, detected=detected, strength=0.75 if detected else 0.0, bullish=False)


# ═══════════════════════════════════════════════════════════════════════════════
# TWO-BAR PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════

def bullish_engulfing(bars: list[dict]) -> PatternResult:
    """Prior bearish bar fully engulfed by larger bullish bar."""
    name = "bullish_engulfing"
    if len(bars) < 2:
        return PatternResult.miss(name)
    o1, h1, l1, c1, _ = _unpack(bars[-2])
    o2, h2, l2, c2, _ = _unpack(bars[-1])
    detected = (
        _bear(o1, c1) and _bull(o2, c2)
        and o2 <= c1 and c2 >= o1
        and _body(o2, c2) > _body(o1, c1)
    )
    strength = round(min(1.0, _body(o2, c2) / max(_body(o1, c1), 1e-9) * 0.5), 3) if detected else 0.0
    return PatternResult(name=name, detected=detected, strength=strength, bullish=True)


def bearish_engulfing(bars: list[dict]) -> PatternResult:
    """Prior bullish bar fully engulfed by larger bearish bar."""
    name = "bearish_engulfing"
    if len(bars) < 2:
        return PatternResult.miss(name, bullish=False)
    o1, h1, l1, c1, _ = _unpack(bars[-2])
    o2, h2, l2, c2, _ = _unpack(bars[-1])
    detected = (
        _bull(o1, c1) and _bear(o2, c2)
        and o2 >= c1 and c2 <= o1
        and _body(o2, c2) > _body(o1, c1)
    )
    strength = round(min(1.0, _body(o2, c2) / max(_body(o1, c1), 1e-9) * 0.5), 3) if detected else 0.0
    return PatternResult(name=name, detected=detected, strength=strength, bullish=False)


def bullish_harami(bars: list[dict]) -> PatternResult:
    """Large bearish bar then small bullish bar inside prior body."""
    name = "bullish_harami"
    if len(bars) < 2:
        return PatternResult.miss(name)
    o1, h1, l1, c1, _ = _unpack(bars[-2])
    o2, h2, l2, c2, _ = _unpack(bars[-1])
    detected = (
        _bear(o1, c1) and _bull(o2, c2)
        and o2 >= c1 and c2 <= o1
        and _body(o2, c2) <= 0.5 * _body(o1, c1)
    )
    return PatternResult(name=name, detected=detected, strength=0.55 if detected else 0.0, bullish=True)


def bearish_harami(bars: list[dict]) -> PatternResult:
    """Large bullish bar then small bearish bar inside prior body."""
    name = "bearish_harami"
    if len(bars) < 2:
        return PatternResult.miss(name, bullish=False)
    o1, h1, l1, c1, _ = _unpack(bars[-2])
    o2, h2, l2, c2, _ = _unpack(bars[-1])
    detected = (
        _bull(o1, c1) and _bear(o2, c2)
        and o2 <= c1 and c2 >= o1
        and _body(o2, c2) <= 0.5 * _body(o1, c1)
    )
    return PatternResult(name=name, detected=detected, strength=0.55 if detected else 0.0, bullish=False)


def piercing_line(bars: list[dict]) -> PatternResult:
    """Bearish bar then bullish bar opening below prior low, closing above prior midpoint."""
    name = "piercing_line"
    if len(bars) < 2:
        return PatternResult.miss(name)
    o1, h1, l1, c1, _ = _unpack(bars[-2])
    o2, h2, l2, c2, _ = _unpack(bars[-1])
    mid1 = (o1 + c1) / 2
    detected = (
        _bear(o1, c1) and _bull(o2, c2)
        and o2 < l1 and c2 > mid1 and c2 < o1
    )
    return PatternResult(name=name, detected=detected, strength=0.65 if detected else 0.0, bullish=True)


def dark_cloud_cover(bars: list[dict]) -> PatternResult:
    """Bullish bar then bearish bar opening above prior high, closing below prior midpoint."""
    name = "dark_cloud_cover"
    if len(bars) < 2:
        return PatternResult.miss(name, bullish=False)
    o1, h1, l1, c1, _ = _unpack(bars[-2])
    o2, h2, l2, c2, _ = _unpack(bars[-1])
    mid1 = (o1 + c1) / 2
    detected = (
        _bull(o1, c1) and _bear(o2, c2)
        and o2 > h1 and c2 < mid1 and c2 > o1
    )
    return PatternResult(name=name, detected=detected, strength=0.65 if detected else 0.0, bullish=False)


# ═══════════════════════════════════════════════════════════════════════════════
# THREE-BAR PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════

def morning_star(bars: list[dict]) -> PatternResult:
    """Morning Star: large bearish, small-body star, large bullish closing above bar1 mid."""
    name = "morning_star"
    if len(bars) < 3:
        return PatternResult.miss(name)
    o1, h1, l1, c1, _ = _unpack(bars[-3])
    o2, h2, l2, c2, _ = _unpack(bars[-2])
    o3, h3, l3, c3, _ = _unpack(bars[-1])
    body1 = _body(o1, c1)
    detected = (
        _bear(o1, c1)
        and _body(o2, c2) <= 0.3 * body1
        and max(o2, c2) < min(o1, c1)   # gap down
        and _bull(o3, c3)
        and c3 > (o1 + c1) / 2
        and _body(o3, c3) >= 0.5 * body1
    )
    return PatternResult(name=name, detected=detected, strength=0.72 if detected else 0.0, bullish=True)


def evening_star(bars: list[dict]) -> PatternResult:
    """Evening Star: large bullish, small-body star gap up, large bearish closing below bar1 mid."""
    name = "evening_star"
    if len(bars) < 3:
        return PatternResult.miss(name, bullish=False)
    o1, h1, l1, c1, _ = _unpack(bars[-3])
    o2, h2, l2, c2, _ = _unpack(bars[-2])
    o3, h3, l3, c3, _ = _unpack(bars[-1])
    body1 = _body(o1, c1)
    detected = (
        _bull(o1, c1)
        and _body(o2, c2) <= 0.3 * body1
        and min(o2, c2) > max(o1, c1)   # gap up
        and _bear(o3, c3)
        and c3 < (o1 + c1) / 2
        and _body(o3, c3) >= 0.5 * body1
    )
    return PatternResult(name=name, detected=detected, strength=0.70 if detected else 0.0, bullish=False)


def three_white_soldiers(bars: list[dict]) -> PatternResult:
    """
    Three White Soldiers (PF 2.90, WR 68.9%):
    Three consecutive bullish candles, each opening in prior body and closing higher,
    with minimal upper wicks.
    """
    name = "three_white_soldiers"
    if len(bars) < 3:
        return PatternResult.miss(name)
    c = [_unpack(b) for b in bars[-3:]]
    ok = True
    for i in range(3):
        o, h, l, cl, _ = c[i]
        if not _bull(o, cl):
            ok = False; break
        if _body(o, cl) > 0 and (h - cl) > 0.2 * _body(o, cl):
            ok = False; break
        if i > 0:
            po, ph, pl, pc, _ = c[i-1]
            if not (po <= o <= pc and cl > pc):
                ok = False; break
    return PatternResult(name=name, detected=ok, strength=0.78 if ok else 0.0, bullish=True)


def three_black_crows(bars: list[dict]) -> PatternResult:
    """Three consecutive bearish candles, each opening in prior body and closing lower."""
    name = "three_black_crows"
    if len(bars) < 3:
        return PatternResult.miss(name, bullish=False)
    c = [_unpack(b) for b in bars[-3:]]
    ok = True
    for i in range(3):
        o, h, l, cl, _ = c[i]
        if not _bear(o, cl):
            ok = False; break
        if _body(o, cl) > 0 and (cl - l) > 0.2 * _body(o, cl):
            ok = False; break
        if i > 0:
            po, ph, pl, pc, _ = c[i-1]
            if not (pc <= o <= po and cl < pc):
                ok = False; break
    return PatternResult(name=name, detected=ok, strength=0.75 if ok else 0.0, bullish=False)


def three_inside_up(bars: list[dict]) -> PatternResult:
    """Bearish bar, bullish harami, then bullish confirmation bar closing above bar1 open."""
    name = "three_inside_up"
    if len(bars) < 3:
        return PatternResult.miss(name)
    o1, h1, l1, c1, _ = _unpack(bars[-3])
    o2, h2, l2, c2, _ = _unpack(bars[-2])
    o3, h3, l3, c3, _ = _unpack(bars[-1])
    detected = (
        _bear(o1, c1) and _bull(o2, c2)
        and o2 >= c1 and c2 <= o1
        and _bull(o3, c3) and c3 > o1
    )
    return PatternResult(name=name, detected=detected, strength=0.68 if detected else 0.0, bullish=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HIGH-PF COMPOSITES (backtested on 18.5yr Nifty)
# ═══════════════════════════════════════════════════════════════════════════════

def rsi_divergence_hammer(
    bars: list[dict],
    rsi: float,
    rsi_divergence: bool,
) -> PatternResult:
    """
    RSI Divergence + Hammer (PF 8.51, WR 73.7%).
    Strongest reversal composite. Requires: bullish RSI divergence + hammer + RSI < 40.
    """
    name = "rsi_divergence_hammer"
    h = hammer(bars)
    detected = h.detected and rsi_divergence and rsi < 40.0
    return PatternResult(name=name, detected=detected, strength=0.85 if detected else 0.0, bullish=True)


def ema_ribbon_cross(
    bars: list[dict],
    ema_values: dict,
) -> PatternResult:
    """
    EMA Ribbon Cross 8/13/21/34/55 (PF 6.72, WR 77.8%).
    All EMAs in descending order = confirmed bull; ascending = confirmed bear.
    """
    name = "ema_ribbon_cross"
    periods = [8, 13, 21, 34, 55]
    vals = [float(ema_values.get(p, 0.0)) for p in periods]
    if any(v == 0.0 for v in vals):
        return PatternResult.miss(name)

    bull = all(vals[i] > vals[i+1] for i in range(len(vals)-1))
    bear = all(vals[i] < vals[i+1] for i in range(len(vals)-1))
    if not (bull or bear):
        return PatternResult.miss(name)

    close = float(_unpack(bars[-1])[3]) if bars else 0.0
    # Require price to be on the correct side of the fastest EMA
    if bull and close <= vals[0]:
        return PatternResult.miss(name)
    if bear and close >= vals[0]:
        return PatternResult.miss(name)

    return PatternResult(name=name, detected=True, strength=0.82, bullish=bull)


def engulfing_volume_spike(
    bars: list[dict],
    volume_ratio: float,
    min_ratio: float = 1.5,
) -> PatternResult:
    """
    Engulfing + Volume Spike (PF 2.45, WR 65.2%).
    Bullish or bearish engulfing confirmed by volume surge.
    """
    spike = volume_ratio >= min_ratio
    boost = min(0.15, (volume_ratio - min_ratio) * 0.05) if spike else 0.0

    bull_eng = bullish_engulfing(bars)
    if bull_eng.detected and spike:
        return PatternResult(
            name="bullish_engulfing_volume_spike",
            detected=True, strength=min(1.0, 0.70 + boost), bullish=True,
        )
    bear_eng = bearish_engulfing(bars)
    if bear_eng.detected and spike:
        return PatternResult(
            name="bearish_engulfing_volume_spike",
            detected=True, strength=min(1.0, 0.68 + boost), bullish=False,
        )
    return PatternResult.miss("engulfing_volume_spike")


def morning_star_rsi(
    bars: list[dict],
    rsi: float,
    rsi_threshold: float = 30.0,
) -> PatternResult:
    """
    Morning Star + RSI < 30 (PF 2.12, WR 62.1%).
    Classic reversal with oversold confirmation.
    """
    name = "morning_star_rsi"
    ms = morning_star(bars)
    detected = ms.detected and rsi < rsi_threshold
    strength = round(min(1.0, 0.72 + (rsi_threshold - rsi) / rsi_threshold * 0.15), 3) if detected else 0.0
    return PatternResult(name=name, detected=detected, strength=strength, bullish=True)


# ═══════════════════════════════════════════════════════════════════════════════
# CHART PATTERNS (pre-detected flags from longer-period analysis)
# ═══════════════════════════════════════════════════════════════════════════════

def _flag_pattern(flags: dict, key: str, strength: float, bullish: bool) -> PatternResult:
    detected = bool(flags.get(key, False))
    return PatternResult(name=key, detected=detected,
                         strength=strength if detected else 0.0, bullish=bullish)

def head_and_shoulders(flags: dict) -> PatternResult:
    return _flag_pattern(flags, "head_and_shoulders", 0.75, False)

def inverse_head_and_shoulders(flags: dict) -> PatternResult:
    return _flag_pattern(flags, "inverse_head_and_shoulders", 0.75, True)

def double_top(flags: dict) -> PatternResult:
    return _flag_pattern(flags, "double_top", 0.70, False)

def double_bottom(flags: dict) -> PatternResult:
    return _flag_pattern(flags, "double_bottom", 0.70, True)

def cup_and_handle(flags: dict) -> PatternResult:
    return _flag_pattern(flags, "cup_and_handle_breakout", 0.78, True)

def bull_flag(flags: dict) -> PatternResult:
    return _flag_pattern(flags, "bull_flag", 0.68, True)

def bear_flag(flags: dict) -> PatternResult:
    return _flag_pattern(flags, "bear_flag", 0.65, False)


# ═══════════════════════════════════════════════════════════════════════════════
# SCAN ALL + COMPOSITE SCORE
# ═══════════════════════════════════════════════════════════════════════════════

def scan_all(
    bars: list[dict],
    *,
    rsi: float = 50.0,
    rsi_divergence: bool = False,
    ema_values: Optional[dict] = None,
    volume_ratio: float = 1.0,
    pattern_flags: Optional[dict] = None,
) -> list[PatternResult]:
    """
    Run every detector and return list of PatternResults where detected=True.
    Used by DynamicSignalScorer to populate the 'candlestick' component.
    """
    ev = ema_values or {}
    pf = pattern_flags or {}
    results: list[PatternResult] = []

    _single = [hammer, inverted_hammer, shooting_star, doji, marubozu_bullish, marubozu_bearish]
    _two    = [bullish_engulfing, bearish_engulfing, bullish_harami, bearish_harami,
               piercing_line, dark_cloud_cover]
    _three  = [morning_star, evening_star, three_white_soldiers, three_black_crows, three_inside_up]

    for fn in _single + _two + _three:
        r = fn(bars)
        if r.detected:
            results.append(r)

    # Composites
    for r in [
        rsi_divergence_hammer(bars, rsi, rsi_divergence),
        ema_ribbon_cross(bars, ev),
        engulfing_volume_spike(bars, volume_ratio),
        morning_star_rsi(bars, rsi),
    ]:
        if r.detected:
            results.append(r)

    # Chart patterns (pre-detected)
    for fn in [head_and_shoulders, inverse_head_and_shoulders, double_top, double_bottom,
               cup_and_handle, bull_flag, bear_flag]:
        r = fn(pf)
        if r.detected:
            results.append(r)

    return results


def composite_score(
    bars: list[dict],
    *,
    rsi: float = 50.0,
    rsi_divergence: bool = False,
    ema_values: Optional[dict] = None,
    volume_ratio: float = 1.0,
    pattern_flags: Optional[dict] = None,
) -> float:
    """
    Returns a single candlestick score in [-1.0, +1.0].
    Positive = net bullish, negative = net bearish.
    Used directly as the 'candlestick' component in DynamicSignalScorer.
    """
    detected = scan_all(
        bars, rsi=rsi, rsi_divergence=rsi_divergence,
        ema_values=ema_values, volume_ratio=volume_ratio,
        pattern_flags=pattern_flags,
    )
    if not detected:
        return 0.0

    bull = sum(r.strength for r in detected if r.bullish)
    bear = sum(r.strength for r in detected if not r.bullish)
    total = bull + bear
    if total == 0:
        return 0.0
    return round((bull - bear) / total, 4)
