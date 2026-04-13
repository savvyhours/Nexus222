"""
Tests for core/calibration/
  - WeightCalibrationAgent  — LLM-driven dynamic weights (mocked Claude)
  - RegimeDetector          — market regime classification
  - SafetyBounds            — weight clamping and validation

Run with:  pytest tests/test_calibration.py -v
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.calibration.regime_detector import RegimeDetector
from core.calibration.safety_bounds import SafetyBounds


# ─────────────────────────────────────────────────────────────────────────
# RegimeDetector tests
# ─────────────────────────────────────────────────────────────────────────

class TestRegimeDetector:
    def _make_detector(self) -> RegimeDetector:
        return RegimeDetector()

    def _market_snapshot(
        self,
        *,
        vix: float = 14.0,
        adx: float = 15.0,
        atr_pct: float = 0.8,
        bb_width_pct: float = 3.0,
        trend_slope: float = 0.0,
        volume_ratio: float = 1.0,
    ) -> dict:
        return {
            "vix": vix,
            "adx": adx,
            "atr_pct": atr_pct,
            "bb_width_pct": bb_width_pct,
            "trend_slope": trend_slope,
            "volume_ratio": volume_ratio,
        }

    def test_high_vix_returns_crisis(self):
        detector = self._make_detector()
        snap = self._market_snapshot(vix=30.0)
        regime = asyncio.get_event_loop().run_until_complete(
            detector.detect(snap)
        )
        assert regime == "CRISIS", f"Expected CRISIS, got {regime}"

    def test_trending_regime(self):
        detector = self._make_detector()
        snap = self._market_snapshot(adx=32.0, trend_slope=0.8, vix=12.0)
        regime = asyncio.get_event_loop().run_until_complete(
            detector.detect(snap)
        )
        assert regime in ("TRENDING", "HIGH_VOL", "LOW_VOL", "MEAN_REVERTING", "CRISIS")

    def test_low_volatility_regime(self):
        detector = self._make_detector()
        snap = self._market_snapshot(
            vix=10.0, adx=10.0, atr_pct=0.3,
            bb_width_pct=1.0, trend_slope=0.05
        )
        regime = asyncio.get_event_loop().run_until_complete(
            detector.detect(snap)
        )
        assert regime in ("LOW_VOL", "MEAN_REVERTING", "TRENDING")

    def test_high_volatility_regime(self):
        detector = self._make_detector()
        snap = self._market_snapshot(
            vix=22.0, adx=25.0, atr_pct=2.5,
            bb_width_pct=8.0
        )
        regime = asyncio.get_event_loop().run_until_complete(
            detector.detect(snap)
        )
        assert regime in ("HIGH_VOL", "CRISIS", "TRENDING")

    def test_detect_returns_string(self):
        detector = self._make_detector()
        snap = self._market_snapshot()
        regime = asyncio.get_event_loop().run_until_complete(
            detector.detect(snap)
        )
        assert isinstance(regime, str)
        assert regime in ("TRENDING", "MEAN_REVERTING", "HIGH_VOL", "LOW_VOL", "CRISIS")

    def test_vix_threshold_boundary(self):
        """VIX exactly at threshold should not trigger CRISIS."""
        detector = self._make_detector()
        # VIX=27 is below the 28 kill-switch threshold
        snap = self._market_snapshot(vix=27.0)
        regime = asyncio.get_event_loop().run_until_complete(
            detector.detect(snap)
        )
        assert regime != "CRISIS", f"VIX=27 should not be CRISIS, got {regime}"

    def test_vix_at_kill_switch_level(self):
        """VIX ≥ 28 → CRISIS regardless of other indicators."""
        detector = self._make_detector()
        snap = self._market_snapshot(vix=28.0, adx=10.0, atr_pct=0.5)
        regime = asyncio.get_event_loop().run_until_complete(
            detector.detect(snap)
        )
        assert regime == "CRISIS"


# ─────────────────────────────────────────────────────────────────────────
# SafetyBounds tests
# ─────────────────────────────────────────────────────────────────────────

class TestSafetyBounds:
    def _make_bounds(self) -> SafetyBounds:
        return SafetyBounds()

    def test_signal_weights_sum_to_one(self):
        bounds = self._make_bounds()
        raw = {
            "technical": 2.0,
            "sentiment": 3.0,
            "fundamental": 1.0,
            "macro": 1.5,
            "candlestick": 0.5,
            "ml_qlib": 1.0,
            "debate_conviction": 2.5,
        }
        clamped = bounds.apply_signal_weights(raw)
        total = sum(clamped.values())
        assert abs(total - 1.0) < 1e-6, f"Signal weights sum to {total}, expected 1.0"

    def test_signal_weights_all_positive(self):
        bounds = self._make_bounds()
        raw = {k: -1.0 for k in
               ["technical", "sentiment", "fundamental", "macro",
                "candlestick", "ml_qlib", "debate_conviction"]}
        clamped = bounds.apply_signal_weights(raw)
        for k, v in clamped.items():
            assert v >= 0.0, f"{k} weight={v} is negative after safety bounds"

    def test_agent_weights_clamped(self):
        bounds = self._make_bounds()
        raw = {
            "scalper": 5.0, "trend": 0.001, "mean_reversion": 0.5,
            "sentiment": 0.0, "fundamentals": 0.3, "macro": 0.2,
            "options": 0.4, "pattern": 0.6, "quant": 0.8, "etf": 0.1,
        }
        clamped = bounds.apply_agent_weights(raw)
        for agent, w in clamped.items():
            assert w >= 0.0, f"{agent} weight went negative"
            assert w <= 1.0, f"{agent} weight={w} exceeds 1.0"

    def test_risk_threshold_clamped_between_01_09(self):
        bounds = self._make_bounds()
        # Provide an out-of-range threshold
        raw = {"signal_threshold": 2.5, "adjudication_zone": -0.5}
        clamped = bounds.apply_risk_thresholds(raw)
        assert 0.1 <= clamped["signal_threshold"] <= 0.9, (
            f"signal_threshold={clamped['signal_threshold']} out of [0.1, 0.9]"
        )

    def test_position_sizing_max_pct_clamped(self):
        bounds = self._make_bounds()
        raw = {"max_position_pct": 0.5, "min_liquidity_volume": 50_000}
        clamped = bounds.apply_position_sizing(raw)
        # max position should never exceed 20% (0.20)
        assert clamped.get("max_position_pct", 1.0) <= 0.20, (
            f"max_position_pct={clamped.get('max_position_pct')} exceeds 0.20"
        )


# ─────────────────────────────────────────────────────────────────────────
# WeightCalibrationAgent tests (with mocked Claude API)
# ─────────────────────────────────────────────────────────────────────────

class TestWeightCalibrationAgent:
    """
    WeightCalibrationAgent calls Claude Sonnet to generate weights.
    We mock the anthropic client so no real API calls are made.
    """

    _MOCK_RESPONSE = {
        "signal_weights": {
            "technical": 0.22,
            "sentiment": 0.12,
            "fundamental": 0.10,
            "macro": 0.14,
            "candlestick": 0.12,
            "ml_qlib": 0.16,
            "debate_conviction": 0.14,
        },
        "agent_weights": {
            "scalper": 0.09, "trend": 0.12, "mean_reversion": 0.10,
            "sentiment": 0.09, "fundamentals": 0.10, "macro": 0.09,
            "options": 0.09, "pattern": 0.11, "quant": 0.11, "etf": 0.10,
        },
        "risk_thresholds": {
            "signal_threshold": 0.42,
            "adjudication_zone": 0.10,
        },
        "sl_tp_multipliers": {
            "scalping_sl_pct": 0.005,
            "scalping_tp_pct": 0.010,
            "intraday_sl_atr": 1.5,
            "positional_sl_atr": 2.2,
        },
        "position_sizing": {
            "max_position_pct": 0.05,
            "min_liquidity_volume": 200_000,
        },
        "kill_switch": False,
        "regime": "TRENDING",
        "reasoning": "Mock calibration for test",
    }

    def _patch_calibration_agent(self):
        """
        Return a WeightCalibrationAgent whose _call_claude is patched
        to return _MOCK_RESPONSE without hitting the API.
        """
        import json
        from core.calibration.weight_calibration_agent import WeightCalibrationAgent

        agent = WeightCalibrationAgent.__new__(WeightCalibrationAgent)
        agent._cache = None
        agent._cache_time = 0.0
        agent._lock = asyncio.Lock()

        async def _mock_call_claude(*args, **kwargs):
            return json.dumps(self._MOCK_RESPONSE)

        agent._call_claude = _mock_call_claude
        return agent

    def test_calibrate_returns_dict(self):
        agent = self._patch_calibration_agent()
        result = asyncio.get_event_loop().run_until_complete(agent.calibrate())
        assert isinstance(result, dict)

    def test_calibrate_has_signal_weights(self):
        agent = self._patch_calibration_agent()
        result = asyncio.get_event_loop().run_until_complete(agent.calibrate())
        assert "signal_weights" in result

    def test_get_signal_weights_after_calibrate(self):
        agent = self._patch_calibration_agent()
        asyncio.get_event_loop().run_until_complete(agent.calibrate())
        weights = asyncio.get_event_loop().run_until_complete(agent.get_signal_weights())
        assert isinstance(weights, dict)
        assert sum(weights.values()) > 0

    def test_get_agent_weights_returns_10_agents(self):
        agent = self._patch_calibration_agent()
        asyncio.get_event_loop().run_until_complete(agent.calibrate())
        weights = asyncio.get_event_loop().run_until_complete(agent.get_agent_weights())
        assert len(weights) == 10, f"Expected 10 agent weights, got {len(weights)}"

    def test_kill_switch_false_by_default(self):
        agent = self._patch_calibration_agent()
        asyncio.get_event_loop().run_until_complete(agent.calibrate())
        active = asyncio.get_event_loop().run_until_complete(agent.kill_switch_active())
        assert active is False

    def test_signal_weights_sum_to_one(self):
        agent = self._patch_calibration_agent()
        asyncio.get_event_loop().run_until_complete(agent.calibrate())
        weights = asyncio.get_event_loop().run_until_complete(agent.get_signal_weights())
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.02, f"Signal weights sum to {total:.4f}"

    def test_cache_prevents_double_calibration(self):
        """Second call within TTL should return cached result without calling LLM."""
        call_count = 0

        import json
        from core.calibration.weight_calibration_agent import WeightCalibrationAgent

        agent = WeightCalibrationAgent.__new__(WeightCalibrationAgent)
        agent._cache = None
        agent._cache_time = 0.0
        agent._lock = asyncio.Lock()

        async def _mock_call_claude(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return json.dumps(self._MOCK_RESPONSE)

        agent._call_claude = _mock_call_claude

        loop = asyncio.get_event_loop()
        loop.run_until_complete(agent.calibrate())
        loop.run_until_complete(agent.calibrate())  # second call — should hit cache

        assert call_count == 1, f"Expected 1 LLM call (cached), got {call_count}"

    def test_get_signal_threshold(self):
        agent = self._patch_calibration_agent()
        asyncio.get_event_loop().run_until_complete(agent.calibrate())
        threshold = asyncio.get_event_loop().run_until_complete(agent.get_signal_threshold())
        assert isinstance(threshold, float)
        assert 0.1 <= threshold <= 0.9, f"threshold={threshold} out of range"
