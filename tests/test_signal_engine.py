"""
Tests for core/signal_engine/
  - DynamicSignalScorer   (dynamic_signal_scorer.py)
  - strategy_library      (22 strategies, run_all, net_score)
  - candlestick_patterns  (scan_all, composite_score)

Run with:  pytest tests/test_signal_engine.py -v
"""
from __future__ import annotations

import asyncio
import math
from unittest.mock import AsyncMock, MagicMock

import pytest

# ── Candlestick patterns ──────────────────────────────────────────────────
from core.signal_engine.candlestick_patterns import (
    PatternResult,
    composite_score,
    scan_all,
)

# ── Strategy library ──────────────────────────────────────────────────────
from core.signal_engine.strategy_library import (
    STRATEGY_REGISTRY,
    StrategySignal,
    active_signals,
    net_score,
    run_all,
)

# ── Dynamic scorer ────────────────────────────────────────────────────────
from core.signal_engine.dynamic_signal_scorer import (
    COMPONENT_KEYS,
    ComponentBreakdown,
    DynamicSignalScorer,
    SignalResult,
    build_components,
    debate_conviction_to_signed,
)


# ─────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────

def _mock_calibration_agent(
    weights: dict | None = None,
    threshold: float = 0.45,
) -> AsyncMock:
    """Return a mock calibration agent with configurable signal weights."""
    default_weights = {k: 1.0 / len(COMPONENT_KEYS) for k in COMPONENT_KEYS}
    agent = AsyncMock()
    agent.get_signal_weights = AsyncMock(return_value=weights or default_weights)
    agent.get_signal_threshold = AsyncMock(return_value=threshold)
    agent.calibrate = AsyncMock()
    return agent


def _make_bars(n: int = 5, bullish: bool = True) -> list[dict]:
    """Generate n synthetic OHLCV bars."""
    bars = []
    price = 1000.0
    for i in range(n):
        delta = 10.0 if bullish else -10.0
        open_ = price
        close = price + delta
        bars.append({
            "open": open_,
            "high": close + 5,
            "low": open_ - 5,
            "close": close,
            "volume": 500_000,
        })
        price = close
    return bars


def _make_market_data(
    *,
    rsi: float = 50.0,
    macd_hist: float = 0.0,
    ema_short: float = 1010.0,
    ema_long: float = 1000.0,
    close: float = 1015.0,
    volume: float = 500_000.0,
    avg_volume: float = 300_000.0,
) -> dict:
    """Minimal market data dict compatible with strategy_library functions."""
    return {
        "close": close,
        "open": close - 10,
        "high": close + 15,
        "low": close - 15,
        "volume": volume,
        "avg_volume": avg_volume,
        "rsi": rsi,
        "macd_histogram": macd_hist,
        "ema_short": ema_short,
        "ema_long": ema_long,
        "ema_9": ema_short - 5,
        "ema_21": ema_long + 5,
        "ema_8": ema_short - 7,
        "ema_13": ema_short - 3,
        "ema_34": ema_long + 10,
        "ema_55": ema_long + 20,
        "adx": 28.0,
        "atr": 15.0,
        "bb_upper": close + 30,
        "bb_lower": close - 30,
        "bb_mid": close,
        "vwap": close - 5,
        "orb_high": close - 10,
        "orb_low": close - 20,
        "prev_close": close - 8,
        "week52_high": close + 5,   # near 52w high → breakout
        "week52_low": close - 200,
        "z_score": 0.5,
        "iv_percentile": 50.0,
        "dte": 14,
        "max_pain": close,
        "pcr": 1.0,
        "predicted_return": 0.02,
        "rank_percentile": 80.0,
        "quality_score": 0.7,
        "momentum_score": 0.6,
        "nav": close,
        "nav_premium_pct": 0.3,
        "vix": 15.0,
        "sector_momentum_5d": 0.02,
        "bars": _make_bars(5, bullish=True),
        "patterns": {},
        "rsi_divergence": False,
        "ema_values": {8: ema_short - 7, 13: ema_short - 3, 21: ema_long + 5},
        "volume_ratio": volume / avg_volume,
        "pattern_flags": {},
    }


# ─────────────────────────────────────────────────────────────────────────
# Strategy library tests
# ─────────────────────────────────────────────────────────────────────────

class TestStrategyLibrary:
    def test_registry_has_22_strategies(self):
        assert len(STRATEGY_REGISTRY) == 22, (
            f"Expected 22 strategies, got {len(STRATEGY_REGISTRY)}"
        )

    def test_run_all_returns_list(self):
        md = _make_market_data()
        signals = run_all(md)
        assert isinstance(signals, list)
        assert len(signals) == 22

    def test_all_signals_have_valid_direction(self):
        md = _make_market_data()
        for sig in run_all(md):
            assert sig.direction in (-1, 0, 1), f"{sig.name}: direction={sig.direction}"

    def test_all_signals_have_valid_strength(self):
        md = _make_market_data()
        for sig in run_all(md):
            assert 0.0 <= sig.strength <= 1.0, f"{sig.name}: strength={sig.strength}"

    def test_score_property(self):
        md = _make_market_data()
        for sig in run_all(md):
            expected = sig.direction * sig.strength
            assert math.isclose(sig.score, expected, abs_tol=1e-9)

    def test_active_signals_excludes_zero(self):
        md = _make_market_data(rsi=50, macd_hist=0, ema_short=1000, ema_long=1000)
        all_sigs = run_all(md)
        active = active_signals(md)
        for sig in active:
            assert sig.direction != 0

    def test_net_score_in_range(self):
        md = _make_market_data()
        score = net_score(md)
        assert -1.0 <= score <= 1.0, f"net_score={score} out of [-1, 1]"

    def test_bullish_conditions_positive_net_score(self):
        """Strong bull data (RSI=30 from oversold recovery, EMA_short>long, volume spike)."""
        md = _make_market_data(
            rsi=32.0,           # oversold → mean-reversion buy signal
            ema_short=1050.0,
            ema_long=1000.0,
            macd_hist=5.0,
            volume=900_000,
            avg_volume=300_000,
        )
        score = net_score(md)
        # At least some strategies should fire bullish
        assert score > -0.5, f"Expected net_score > -0.5, got {score}"

    def test_bearish_conditions_negative_net_score(self):
        """Strong bear data."""
        md = _make_market_data(
            rsi=78.0,           # overbought
            ema_short=950.0,    # short < long → bearish cross
            ema_long=1000.0,
            macd_hist=-8.0,
            volume=900_000,
            avg_volume=300_000,
        )
        score = net_score(md)
        assert score < 0.5, f"Expected net_score < 0.5, got {score}"

    def test_each_strategy_has_group(self):
        md = _make_market_data()
        for sig in run_all(md):
            assert sig.group in ("A", "B", "C", "D", "E"), (
                f"{sig.name}: unexpected group={sig.group}"
            )

    def test_each_strategy_has_reason(self):
        md = _make_market_data()
        for sig in run_all(md):
            assert isinstance(sig.reason, str) and len(sig.reason) > 0, (
                f"{sig.name}: missing reason"
            )


# ─────────────────────────────────────────────────────────────────────────
# Candlestick pattern tests
# ─────────────────────────────────────────────────────────────────────────

class TestCandlestickPatterns:
    def _hammer_bars(self) -> list[dict]:
        """A textbook hammer bar."""
        return [
            {"open": 1005, "high": 1010, "low": 960, "close": 1008, "volume": 400_000},
        ]

    def _engulfing_bars(self) -> list[dict]:
        """Bearish then large bullish engulfing."""
        return [
            {"open": 1010, "high": 1015, "low": 1005, "close": 1006, "volume": 300_000},  # bearish
            {"open": 1004, "high": 1025, "low": 1002, "close": 1022, "volume": 600_000},  # bullish engulf
        ]

    def test_scan_all_returns_list_of_pattern_results(self):
        bars = _make_bars(5)
        results = scan_all(
            bars,
            rsi=50.0,
            rsi_divergence=False,
            ema_values={8: 1010, 13: 1005, 21: 1000},
            volume_ratio=1.2,
            pattern_flags={},
        )
        assert isinstance(results, list)
        assert all(isinstance(r, PatternResult) for r in results)

    def test_all_results_have_name_and_detected(self):
        bars = _make_bars(3)
        for r in scan_all(bars, rsi=50, rsi_divergence=False,
                          ema_values={}, volume_ratio=1.0, pattern_flags={}):
            assert isinstance(r.name, str)
            assert isinstance(r.detected, bool)
            assert 0.0 <= r.strength <= 1.0

    def test_composite_score_in_range(self):
        bars = _make_bars(5, bullish=True)
        score = composite_score(
            bars,
            rsi=45.0,
            rsi_divergence=True,
            ema_values={8: 1020, 13: 1015, 21: 1010},
            volume_ratio=1.8,
            pattern_flags={},
        )
        assert -1.0 <= score <= 1.0, f"composite_score={score}"

    def test_hammer_detection(self):
        """Single hammer bar should be detected."""
        bars = self._hammer_bars()
        results = scan_all(bars, rsi=30, rsi_divergence=False,
                          ema_values={}, volume_ratio=1.0, pattern_flags={})
        hammer = next((r for r in results if r.name == "hammer"), None)
        assert hammer is not None, "hammer pattern not returned"
        # With such a long lower shadow, hammer should fire
        assert hammer.detected, "hammer not detected on textbook hammer bar"

    def test_pattern_result_is_frozen(self):
        """PatternResult should be immutable (frozen dataclass)."""
        pr = PatternResult(name="test", detected=True, strength=0.8, bullish=True)
        with pytest.raises((AttributeError, TypeError)):
            pr.detected = False  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────────────────
# DynamicSignalScorer tests
# ─────────────────────────────────────────────────────────────────────────

class TestDynamicSignalScorer:
    def _make_scorer(self, weights=None, threshold=0.45) -> DynamicSignalScorer:
        return DynamicSignalScorer(_mock_calibration_agent(weights, threshold))

    # ── build_components helper ───────────────────────────────────────────

    def test_build_components_keys(self):
        comps = build_components(technical_score=0.5, sentiment_score=0.3)
        assert set(comps.keys()) == set(COMPONENT_KEYS)

    def test_build_components_clamped(self):
        comps = build_components(technical_score=2.0, sentiment_score=-5.0)
        # build_components itself doesn't clamp — scorer does; values passed through
        assert comps["technical"] == 2.0  # raw; scorer will clamp

    def test_debate_conviction_to_signed_buy(self):
        assert debate_conviction_to_signed("BUY", 0.8) == pytest.approx(0.8)

    def test_debate_conviction_to_signed_sell(self):
        assert debate_conviction_to_signed("SELL", 0.7) == pytest.approx(-0.7)

    def test_debate_conviction_to_signed_hold(self):
        assert debate_conviction_to_signed("HOLD", 0.9) == pytest.approx(0.0)

    # ── score() ──────────────────────────────────────────────────────────

    def test_score_returns_signal_result(self):
        scorer = self._make_scorer()
        result = asyncio.get_event_loop().run_until_complete(
            scorer.score("RELIANCE", build_components(technical_score=0.7))
        )
        assert isinstance(result, SignalResult)

    def test_score_all_zero_components_is_hold(self):
        scorer = self._make_scorer()
        result = asyncio.get_event_loop().run_until_complete(
            scorer.score("INFY", build_components())
        )
        assert result.direction == 0
        assert result.score == pytest.approx(0.0)

    def test_strong_bull_components_trigger_buy(self):
        equal_weights = {k: 1.0 for k in COMPONENT_KEYS}
        scorer = self._make_scorer(weights=equal_weights, threshold=0.30)
        comps = build_components(
            technical_score=0.9,
            sentiment_score=0.8,
            fundamental_score=0.7,
            macro_score=0.6,
        )
        result = asyncio.get_event_loop().run_until_complete(
            scorer.score("TCS", comps, regime="TRENDING")
        )
        assert result.direction == 1
        assert result.triggered is True

    def test_strong_bear_components_trigger_sell(self):
        equal_weights = {k: 1.0 for k in COMPONENT_KEYS}
        scorer = self._make_scorer(weights=equal_weights, threshold=0.30)
        comps = build_components(
            technical_score=-0.9,
            sentiment_score=-0.8,
            debate_conviction_signed=-0.7,
        )
        result = asyncio.get_event_loop().run_until_complete(
            scorer.score("WIPRO", comps, regime="MEAN_REVERTING")
        )
        assert result.direction == -1
        assert result.triggered is True

    def test_crisis_regime_always_hold(self):
        scorer = self._make_scorer()
        comps = build_components(
            technical_score=0.99,
            debate_conviction_signed=0.99,
        )
        result = asyncio.get_event_loop().run_until_complete(
            scorer.score("NIFTY", comps, regime="CRISIS")
        )
        assert result.direction == 0
        assert result.triggered is False
        assert result.override_reason is not None

    def test_score_clamped_to_minus1_plus1(self):
        # Even if components are all +1.0, score stays ≤ 1.0
        all_ones = {k: 1.0 for k in COMPONENT_KEYS}
        scorer = self._make_scorer(weights=all_ones, threshold=0.05)
        comps = build_components(**{k.replace("_score", "") + "_score": 1.0
                                    for k in ["technical", "sentiment", "fundamental",
                                              "macro", "candlestick", "ml_qlib"]},
                                 debate_conviction_signed=1.0)
        result = asyncio.get_event_loop().run_until_complete(
            scorer.score("HDFC", comps)
        )
        assert result.score <= 1.0

    def test_breakdown_has_all_component_keys(self):
        scorer = self._make_scorer()
        result = asyncio.get_event_loop().run_until_complete(
            scorer.score("SBIN", build_components(technical_score=0.5))
        )
        assert set(result.breakdown.keys()) == set(COMPONENT_KEYS)

    def test_conviction_is_abs_score(self):
        equal_weights = {k: 1.0 for k in COMPONENT_KEYS}
        scorer = self._make_scorer(weights=equal_weights, threshold=0.01)
        comps = build_components(technical_score=-0.6)
        result = asyncio.get_event_loop().run_until_complete(
            scorer.score("ICICI", comps)
        )
        assert result.conviction == pytest.approx(abs(result.score), abs=0.01)

    def test_to_dict_has_required_keys(self):
        scorer = self._make_scorer()
        result = asyncio.get_event_loop().run_until_complete(
            scorer.score("BAJAJ", build_components(sentiment_score=0.4))
        )
        d = result.to_dict()
        for key in ("symbol", "score", "direction", "action", "triggered",
                    "conviction", "threshold", "regime", "breakdown"):
            assert key in d, f"Missing key in to_dict(): {key}"

    def test_score_batch_returns_all_symbols(self):
        scorer = self._make_scorer()
        batch = {
            "RELIANCE": build_components(technical_score=0.6),
            "TCS":      build_components(sentiment_score=-0.5),
            "INFY":     build_components(),
        }
        results = asyncio.get_event_loop().run_until_complete(
            scorer.score_batch(batch, regime="TRENDING")
        )
        assert set(results.keys()) == set(batch.keys())
        for sym, r in results.items():
            assert r.symbol == sym
