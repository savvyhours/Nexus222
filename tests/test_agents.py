"""
Tests for core/agents/ — 10 strategy sub-agents + MasterOrchestrator.

Run with:  pytest tests/test_agents.py -v

All tests use mock calibration agents so no LLM calls are made.
Agent.analyze() is tested for:
  - Return type (list[AgentSignal])
  - Signal action validity (+1/-1/0)
  - Signal strength in [0, 1]
  - weighted_vote = strength * action_numeric
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.agents.base_agent import Action, AgentSignal

# ── Default agent weights (uniform) ──────────────────────────────────────
_DEFAULT_WEIGHTS = {
    "scalper":        0.10,
    "trend":          0.12,
    "mean_reversion": 0.11,
    "sentiment":      0.09,
    "fundamentals":   0.10,
    "macro":          0.09,
    "options":        0.09,
    "pattern":        0.10,
    "quant":          0.10,
    "etf":            0.10,
}

_SIGNAL_THRESHOLDS = {"signal_threshold": 0.45, "adjudication_zone": 0.10}
_SL_TP = {"scalping_sl_pct": 0.005, "scalping_tp_pct": 0.010,
           "intraday_sl_atr": 1.5, "positional_sl_atr": 2.0}
_POSITION = {"max_position_pct": 0.05, "min_liquidity_volume": 100_000}


def _mock_cal() -> AsyncMock:
    cal = AsyncMock()
    cal.get_agent_weights = AsyncMock(return_value=_DEFAULT_WEIGHTS)
    cal.get_signal_weights = AsyncMock(return_value={})
    cal.get_signal_threshold = AsyncMock(return_value=0.45)
    cal.get_risk_thresholds = AsyncMock(return_value=_SIGNAL_THRESHOLDS)
    cal.get_sl_tp_multipliers = AsyncMock(return_value=_SL_TP)
    cal.get_position_sizing = AsyncMock(return_value=_POSITION)
    cal.kill_switch_active = AsyncMock(return_value=False)
    return cal


def _market_data(
    *,
    close: float = 1000.0,
    rsi: float = 50.0,
    volume: float = 500_000,
    avg_volume: float = 300_000,
    ema_short: float = 1010.0,
    ema_long: float = 1000.0,
    vwap: float = 995.0,
    vix: float = 14.0,
    fii_net_crores: float = 1500.0,
    usd_inr: float = 83.0,
) -> dict:
    return {
        "symbol": "TEST",
        "close": close,
        "open": close - 5,
        "high": close + 15,
        "low": close - 15,
        "volume": volume,
        "avg_volume": avg_volume,
        "rsi": rsi,
        "ema_short": ema_short,
        "ema_long": ema_long,
        "ema_9": ema_short - 5,
        "ema_21": ema_long + 5,
        "ema_8": ema_short - 7,
        "ema_13": ema_short - 3,
        "ema_34": ema_long + 10,
        "ema_55": ema_long + 20,
        "macd_histogram": 2.5,
        "adx": 30.0,
        "atr": 12.0,
        "bb_upper": close + 30,
        "bb_lower": close - 30,
        "bb_mid": close,
        "vwap": vwap,
        "orb_high": close - 10,
        "orb_low": close - 20,
        "week52_high": close + 5,
        "week52_low": close - 300,
        "z_score": 0.4,
        "vix": vix,
        "fii_net_crores": fii_net_crores,
        "dii_net_crores": 500.0,
        "usd_inr": usd_inr,
        "crude_brent": 82.0,
        "iv_percentile": 55.0,
        "dte": 12,
        "max_pain": close,
        "pcr": 1.1,
        "predicted_return": 0.025,
        "rank_percentile": 75.0,
        "quality_score": 0.65,
        "momentum_score": 0.7,
        "nav": close,
        "nav_premium_pct": 0.2,
        "sector_momentum_5d": 0.018,
        "pe_ratio": 18.0,
        "roe": 16.0,
        "eps_growth": 17.0,
        "debt_equity": 0.3,
        "qlib_factor_score": 0.7,
        "fii_streak_days": 4,
        "bars": [
            {"open": 990 + i, "high": 1010 + i, "low": 985 + i, "close": 1005 + i, "volume": 450_000}
            for i in range(5)
        ],
        "patterns": {},
        "rsi_divergence": False,
        "ema_values": {"8": ema_short - 7, "13": ema_short - 3, "21": ema_long + 5},
        "volume_ratio": volume / avg_volume,
        "pattern_flags": {},
    }


def _sentiment_data() -> dict:
    return {
        "score": 0.45,
        "velocity": 120.0,
        "sources": ["ET", "Bloomberg"],
        "swing_alert": False,
        "last_updated": "2025-01-01T09:30:00+05:30",
    }


# ── Helper: validate AgentSignal list ─────────────────────────────────────

def _assert_valid_signals(signals: list, agent_name: str):
    assert isinstance(signals, list), f"{agent_name}: should return list"
    for s in signals:
        assert isinstance(s, AgentSignal), f"{agent_name}: not AgentSignal"
        assert s.action in (Action.BUY, Action.SELL, Action.HOLD), \
            f"{agent_name}: invalid action {s.action}"
        assert 0.0 <= s.strength <= 1.0, \
            f"{agent_name}: strength {s.strength} out of [0,1]"
        expected_vote = s.strength * s.action.numeric
        assert abs(s.weighted_vote - expected_vote) < 1e-9, \
            f"{agent_name}: weighted_vote mismatch"


# ─────────────────────────────────────────────────────────────────────────
# Individual agent tests
# ─────────────────────────────────────────────────────────────────────────

class TestScalperAgent:
    def test_returns_valid_signals(self):
        from core.agents.scalper_agent import ScalperAgent
        agent = ScalperAgent(_mock_cal())
        md = _market_data(close=1005, vwap=1000, rsi=35, volume=700_000)
        signals = asyncio.get_event_loop().run_until_complete(
            agent.analyze(md, _sentiment_data())
        )
        _assert_valid_signals(signals, "ScalperAgent")


class TestTrendAgent:
    def test_uptrend_conditions(self):
        from core.agents.trend_agent import TrendFollowerAgent
        agent = TrendFollowerAgent(_mock_cal())
        md = _market_data(ema_short=1050, ema_long=1000, close=1060)
        signals = asyncio.get_event_loop().run_until_complete(
            agent.analyze(md, _sentiment_data())
        )
        _assert_valid_signals(signals, "TrendFollowerAgent")

    def test_downtrend_conditions(self):
        from core.agents.trend_agent import TrendFollowerAgent
        agent = TrendFollowerAgent(_mock_cal())
        md = _market_data(ema_short=950, ema_long=1000, close=940)
        signals = asyncio.get_event_loop().run_until_complete(
            agent.analyze(md, _sentiment_data())
        )
        _assert_valid_signals(signals, "TrendFollowerAgent")


class TestMeanReversionAgent:
    def test_oversold_conditions(self):
        from core.agents.mean_reversion_agent import MeanReversionAgent
        agent = MeanReversionAgent(_mock_cal())
        md = _market_data(rsi=25, close=940)
        md["bb_lower"] = 945  # close below bb_lower
        md["z_score"] = -2.3
        signals = asyncio.get_event_loop().run_until_complete(
            agent.analyze(md, _sentiment_data())
        )
        _assert_valid_signals(signals, "MeanReversionAgent")


class TestSentimentAgent:
    def test_positive_sentiment(self):
        from core.agents.sentiment_agent import SentimentAgent
        agent = SentimentAgent(_mock_cal())
        sd = {"score": 0.72, "velocity": 250.0, "swing_alert": False}
        signals = asyncio.get_event_loop().run_until_complete(
            agent.analyze(_market_data(), sd)
        )
        _assert_valid_signals(signals, "SentimentAgent")

    def test_negative_sentiment(self):
        from core.agents.sentiment_agent import SentimentAgent
        agent = SentimentAgent(_mock_cal())
        sd = {"score": -0.68, "velocity": -180.0, "swing_alert": True}
        signals = asyncio.get_event_loop().run_until_complete(
            agent.analyze(_market_data(), sd)
        )
        _assert_valid_signals(signals, "SentimentAgent")


class TestFundamentalsAgent:
    def test_value_stock_conditions(self):
        from core.agents.fundamentals_agent import FundamentalsAgent
        agent = FundamentalsAgent(_mock_cal())
        md = _market_data()
        md.update({"pe_ratio": 12.0, "roe": 18.0, "eps_growth": 20.0, "qlib_factor_score": 0.75})
        signals = asyncio.get_event_loop().run_until_complete(
            agent.analyze(md, _sentiment_data())
        )
        _assert_valid_signals(signals, "FundamentalsAgent")


class TestMacroAgent:
    def test_fii_bullish_streak(self):
        from core.agents.macro_agent import MacroAgent
        agent = MacroAgent(_mock_cal())
        md = _market_data(fii_net_crores=2000, vix=13)
        md["fii_streak_days"] = 5
        signals = asyncio.get_event_loop().run_until_complete(
            agent.analyze(md, _sentiment_data())
        )
        _assert_valid_signals(signals, "MacroAgent")

    def test_high_vix_returns_signals(self):
        from core.agents.macro_agent import MacroAgent
        agent = MacroAgent(_mock_cal())
        md = _market_data(vix=24.0)
        signals = asyncio.get_event_loop().run_until_complete(
            agent.analyze(md, _sentiment_data())
        )
        _assert_valid_signals(signals, "MacroAgent")


class TestOptionsAgent:
    def test_returns_valid_signals(self):
        from core.agents.options_agent import OptionsAgent
        agent = OptionsAgent(_mock_cal())
        signals = asyncio.get_event_loop().run_until_complete(
            agent.analyze(_market_data(), _sentiment_data())
        )
        _assert_valid_signals(signals, "OptionsAgent")

    def test_iv_crush_conditions(self):
        from core.agents.options_agent import OptionsAgent
        agent = OptionsAgent(_mock_cal())
        md = _market_data()
        md.update({"iv_percentile": 85.0, "dte": 5})
        signals = asyncio.get_event_loop().run_until_complete(
            agent.analyze(md, _sentiment_data())
        )
        _assert_valid_signals(signals, "OptionsAgent")


class TestPatternAgent:
    def test_returns_valid_signals(self):
        from core.agents.pattern_agent import PatternAgent
        agent = PatternAgent(_mock_cal())
        signals = asyncio.get_event_loop().run_until_complete(
            agent.analyze(_market_data(), _sentiment_data())
        )
        _assert_valid_signals(signals, "PatternAgent")


class TestQuantAgent:
    def test_top_decile_returns_buy(self):
        from core.agents.quant_agent import QuantAgent
        agent = QuantAgent(_mock_cal())
        md = _market_data()
        md.update({"rank_percentile": 92.0, "predicted_return": 0.04})
        signals = asyncio.get_event_loop().run_until_complete(
            agent.analyze(md, _sentiment_data())
        )
        _assert_valid_signals(signals, "QuantAgent")

    def test_bottom_decile_returns_sell(self):
        from core.agents.quant_agent import QuantAgent
        agent = QuantAgent(_mock_cal())
        md = _market_data()
        md.update({"rank_percentile": 6.0, "predicted_return": -0.03})
        signals = asyncio.get_event_loop().run_until_complete(
            agent.analyze(md, _sentiment_data())
        )
        _assert_valid_signals(signals, "QuantAgent")


class TestETFAgent:
    def test_nav_discount_buy(self):
        from core.agents.etf_agent import ETFAgent
        agent = ETFAgent(_mock_cal())
        md = _market_data()
        md.update({"nav_premium_pct": -0.7})  # discount → buy
        signals = asyncio.get_event_loop().run_until_complete(
            agent.analyze(md, _sentiment_data())
        )
        _assert_valid_signals(signals, "ETFAgent")


# ─────────────────────────────────────────────────────────────────────────
# Base Agent contract
# ─────────────────────────────────────────────────────────────────────────

class TestBaseAgentContract:
    def test_action_enum_numeric(self):
        assert Action.BUY.numeric == 1
        assert Action.SELL.numeric == -1
        assert Action.HOLD.numeric == 0

    def test_agent_signal_weighted_vote(self):
        sig = AgentSignal(
            agent_name="test",
            action=Action.BUY,
            strength=0.75,
            reason="test",
        )
        assert sig.weighted_vote == pytest.approx(0.75)

    def test_sell_signal_negative_vote(self):
        sig = AgentSignal(
            agent_name="test",
            action=Action.SELL,
            strength=0.60,
            reason="test",
        )
        assert sig.weighted_vote == pytest.approx(-0.60)

    def test_hold_signal_zero_vote(self):
        sig = AgentSignal(
            agent_name="test",
            action=Action.HOLD,
            strength=0.50,
            reason="test",
        )
        assert sig.weighted_vote == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────
# MasterOrchestrator smoke test
# ─────────────────────────────────────────────────────────────────────────

class TestMasterOrchestrator:
    def _make_mock_agent(self, name: str, action: Action, strength: float):
        """Agent that always returns a fixed signal."""
        mock = AsyncMock()
        mock.AGENT_KEY = name
        mock.get_weight = AsyncMock(return_value=_DEFAULT_WEIGHTS.get(name, 0.1))
        mock.analyze = AsyncMock(return_value=[
            AgentSignal(agent_name=name, action=action, strength=strength, reason="mock")
        ])
        return mock

    def test_consensus_buy_signal(self):
        from core.agents.master_orchestrator import MasterOrchestrator

        agents = [
            self._make_mock_agent("scalper", Action.BUY, 0.8),
            self._make_mock_agent("trend", Action.BUY, 0.9),
            self._make_mock_agent("sentiment", Action.BUY, 0.7),
        ]
        cal = _mock_cal()
        cal.get_risk_thresholds = AsyncMock(return_value={
            "signal_threshold": 0.30,
            "adjudication_zone": 0.10,
        })

        orch = MasterOrchestrator(agents=agents, calibration_agent=cal)
        results = asyncio.get_event_loop().run_until_complete(
            orch.run_cycle(
                universe=["RELIANCE"],
                market_data_map={"RELIANCE": _market_data()},
                sentiment_data_map={"RELIANCE": _sentiment_data()},
            )
        )
        assert len(results) == 1
        cs = results[0]
        assert cs.symbol == "RELIANCE"
        assert cs.action == "BUY"
        assert cs.weighted_vote > 0

    def test_no_signal_below_threshold(self):
        from core.agents.master_orchestrator import MasterOrchestrator

        # All HOLD → no consensus
        agents = [
            self._make_mock_agent("scalper", Action.HOLD, 0.1),
            self._make_mock_agent("trend", Action.HOLD, 0.1),
        ]
        cal = _mock_cal()
        cal.get_risk_thresholds = AsyncMock(return_value={
            "signal_threshold": 0.80,
            "adjudication_zone": 0.05,
        })
        orch = MasterOrchestrator(agents=agents, calibration_agent=cal)
        results = asyncio.get_event_loop().run_until_complete(
            orch.run_cycle(
                universe=["TCS"],
                market_data_map={"TCS": _market_data()},
                sentiment_data_map={"TCS": _sentiment_data()},
            )
        )
        assert results == []
