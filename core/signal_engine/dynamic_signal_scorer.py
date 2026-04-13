"""
NEXUS-II v2.1 — DynamicSignalScorer
Layer 6: combines 7 component signals with regime-adaptive weights from WeightCalibrationAgent.

Components
----------
technical        — RSI, MACD, EMA, ADX, VWAP (from TechnicalAnalyst / strategy_library Group A)
sentiment        — NLP news/social score (from SentimentAnalyst)
fundamental      — P/E, ROE, EPS growth, Qlib factor (from FundamentalAnalyst)
macro            — VIX, FII flows, USD/INR, crude (from MacroAnalyst)
candlestick      — composite_score() from candlestick_patterns (Group B)
ml_qlib          — Qlib ML rank percentile (from QuantAgent)
debate_conviction— signed conviction from DebateArena (+ve BUY, -ve SELL)

All weights and thresholds fetched live from WeightCalibrationAgent (15-min TTL).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ComponentBreakdown:
    """Per-component contribution to the final score."""
    raw: float          # raw component value in [-1.0, +1.0]
    weight: float       # dynamic weight from calibration
    contribution: float # raw * weight

    @property
    def pct(self) -> float:
        """Contribution as percentage of total abs score (filled by SignalResult)."""
        return self._pct

    _pct: float = field(default=0.0, init=False, repr=False)


@dataclass
class SignalResult:
    """
    Final signal for one symbol from the scoring engine.

    score        — weighted sum in approx [-1.0, +1.0]
    direction    — +1 BUY | -1 SELL | 0 HOLD
    triggered    — True when |score| >= threshold
    conviction   — |score| normalised to [0, 1]
    threshold    — dynamic threshold used (from calibration)
    regime       — market regime at time of scoring
    breakdown    — per-component contribution
    """
    symbol: str
    score: float
    direction: int          # +1 / -1 / 0
    triggered: bool
    conviction: float       # |score| / 1.0 clamped to [0,1]
    threshold: float
    regime: str
    breakdown: dict[str, ComponentBreakdown]
    override_reason: Optional[str] = None   # populated when hard kill-switch fires

    # --- helpers ---
    @property
    def action(self) -> str:
        return {1: "BUY", -1: "SELL", 0: "HOLD"}[self.direction]

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "score": round(self.score, 4),
            "direction": self.direction,
            "action": self.action,
            "triggered": self.triggered,
            "conviction": round(self.conviction, 4),
            "threshold": round(self.threshold, 4),
            "regime": self.regime,
            "override_reason": self.override_reason,
            "breakdown": {
                k: {
                    "raw": round(v.raw, 4),
                    "weight": round(v.weight, 4),
                    "contribution": round(v.contribution, 4),
                }
                for k, v in self.breakdown.items()
            },
        }


# ---------------------------------------------------------------------------
# Component keys (canonical ordering)
# ---------------------------------------------------------------------------

COMPONENT_KEYS = (
    "technical",
    "sentiment",
    "fundamental",
    "macro",
    "candlestick",
    "ml_qlib",
    "debate_conviction",
)


# ---------------------------------------------------------------------------
# DynamicSignalScorer
# ---------------------------------------------------------------------------

class DynamicSignalScorer:
    """
    Combines the 7 signal components using calibration-agent weights.

    Usage
    -----
        scorer = DynamicSignalScorer(calibration_agent)
        result = await scorer.score(symbol, components, regime)

    Parameters
    ----------
    calibration_agent
        Instance of WeightCalibrationAgent (from core.calibration).
    """

    # Kill-switch: absolute score must exceed this floor even after dynamic weights
    _MINIMUM_SCORE_FLOOR = 0.05

    def __init__(self, calibration_agent) -> None:
        self._cal = calibration_agent

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def score(
        self,
        symbol: str,
        components: dict[str, float],
        regime: str = "UNKNOWN",
    ) -> SignalResult:
        """
        Score one symbol.

        Parameters
        ----------
        symbol      — NSE/BSE ticker, e.g. "RELIANCE"
        components  — dict with keys matching COMPONENT_KEYS; values in [-1.0, +1.0].
                      Missing keys treated as 0.0.
        regime      — market regime string (TRENDING / MEAN_REVERTING / HIGH_VOL /
                      LOW_VOL / CRISIS).  Passed for logging; calibration agent
                      already knows the regime when it sets weights.

        Returns
        -------
        SignalResult
        """
        # --- 1. Fetch dynamic weights & threshold (cached, 15-min TTL) ---
        weights: dict[str, float] = await self._cal.get_signal_weights()
        threshold: float = await self._cal.get_signal_threshold()

        # --- 2. Validate / clamp components ---
        validated = self._validate_components(components)

        # --- 3. Kill switch: if regime is CRISIS return HOLD immediately ---
        if regime == "CRISIS":
            return self._crisis_result(symbol, threshold, regime)

        # --- 4. Weighted sum ---
        breakdown: dict[str, ComponentBreakdown] = {}
        total_score = 0.0
        total_abs_weight = 0.0

        for key in COMPONENT_KEYS:
            raw = validated.get(key, 0.0)
            w = weights.get(key, 0.0)
            contrib = raw * w
            breakdown[key] = ComponentBreakdown(raw=raw, weight=w, contribution=contrib)
            total_score += contrib
            total_abs_weight += abs(w)

        # Normalise so weights don't inflate score beyond [-1, +1]
        if total_abs_weight > 0:
            total_score = total_score / total_abs_weight

        # Clamp to [-1, 1]
        total_score = max(-1.0, min(1.0, total_score))

        # --- 5. Fill percentage breakdown ---
        abs_score = abs(total_score)
        for bd in breakdown.values():
            bd._pct = abs(bd.contribution) / abs_score if abs_score > 1e-9 else 0.0

        # --- 6. Direction & conviction ---
        direction, triggered, conviction = self._classify(total_score, threshold)

        log.debug(
            "DynamicSignalScorer | %s | score=%.4f | threshold=%.4f | "
            "direction=%s | regime=%s",
            symbol, total_score, threshold, direction, regime,
        )

        return SignalResult(
            symbol=symbol,
            score=total_score,
            direction=direction,
            triggered=triggered,
            conviction=conviction,
            threshold=threshold,
            regime=regime,
            breakdown=breakdown,
        )

    async def score_batch(
        self,
        symbols_components: dict[str, dict[str, float]],
        regime: str = "UNKNOWN",
    ) -> dict[str, SignalResult]:
        """
        Score multiple symbols concurrently.

        Parameters
        ----------
        symbols_components — {symbol: components_dict}

        Returns
        -------
        {symbol: SignalResult}
        """
        import asyncio

        tasks = {
            sym: asyncio.create_task(self.score(sym, comps, regime))
            for sym, comps in symbols_components.items()
        }
        results: dict[str, SignalResult] = {}
        for sym, task in tasks.items():
            try:
                results[sym] = await task
            except Exception as exc:
                log.error("DynamicSignalScorer.score_batch | %s | error: %s", sym, exc)
                results[sym] = self._error_result(sym, regime)
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_components(raw: dict[str, float]) -> dict[str, float]:
        """Clamp every component to [-1.0, +1.0] and fill missing keys with 0."""
        out: dict[str, float] = {}
        for key in COMPONENT_KEYS:
            val = float(raw.get(key, 0.0))
            out[key] = max(-1.0, min(1.0, val))
        return out

    @staticmethod
    def _classify(score: float, threshold: float) -> tuple[int, bool, float]:
        """Return (direction, triggered, conviction)."""
        abs_score = abs(score)
        triggered = abs_score >= threshold
        conviction = min(1.0, abs_score)

        if not triggered:
            direction = 0
        else:
            direction = 1 if score > 0 else -1

        return direction, triggered, conviction

    def _crisis_result(self, symbol: str, threshold: float, regime: str) -> SignalResult:
        empty_bd = {
            k: ComponentBreakdown(raw=0.0, weight=0.0, contribution=0.0)
            for k in COMPONENT_KEYS
        }
        return SignalResult(
            symbol=symbol,
            score=0.0,
            direction=0,
            triggered=False,
            conviction=0.0,
            threshold=threshold,
            regime=regime,
            breakdown=empty_bd,
            override_reason="CRISIS regime — kill switch active",
        )

    def _error_result(self, symbol: str, regime: str) -> SignalResult:
        empty_bd = {
            k: ComponentBreakdown(raw=0.0, weight=0.0, contribution=0.0)
            for k in COMPONENT_KEYS
        }
        return SignalResult(
            symbol=symbol,
            score=0.0,
            direction=0,
            triggered=False,
            conviction=0.0,
            threshold=0.5,
            regime=regime,
            breakdown=empty_bd,
            override_reason="Scoring error — defaulting to HOLD",
        )


# ---------------------------------------------------------------------------
# Component builder helpers
# ---------------------------------------------------------------------------

def build_components(
    *,
    technical_score: float = 0.0,
    sentiment_score: float = 0.0,
    fundamental_score: float = 0.0,
    macro_score: float = 0.0,
    candlestick_score: float = 0.0,
    ml_qlib_score: float = 0.0,
    debate_conviction_signed: float = 0.0,
) -> dict[str, float]:
    """
    Convenience builder so callers don't need to remember the exact key names.

    All values in [-1.0, +1.0].
    debate_conviction_signed: positive for BUY verdict, negative for SELL verdict.

    Example
    -------
        comps = build_components(
            technical_score=0.72,
            sentiment_score=0.45,
            debate_conviction_signed=0.68,   # BUY with 0.68 conviction
        )
        result = await scorer.score("RELIANCE", comps, regime="TRENDING")
    """
    return {
        "technical": technical_score,
        "sentiment": sentiment_score,
        "fundamental": fundamental_score,
        "macro": macro_score,
        "candlestick": candlestick_score,
        "ml_qlib": ml_qlib_score,
        "debate_conviction": debate_conviction_signed,
    }


def debate_conviction_to_signed(direction: str, conviction: float) -> float:
    """
    Convert DebateVerdict direction + conviction to a signed score.

    direction  — "BUY" | "SELL" | "HOLD"
    conviction — [0, 1]
    """
    direction = direction.upper()
    if direction == "BUY":
        return abs(conviction)
    if direction == "SELL":
        return -abs(conviction)
    return 0.0
