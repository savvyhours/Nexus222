"""
NEXUS-II — Regime Detector
Classifies the current market into one of five regimes using technical indicators
and volatility data. This is a fast, rule-based classification that runs locally
(no LLM call) to provide context for the WeightCalibrationAgent prompt.

Regimes:
    TRENDING        — ADX > 25, clear directional move
    MEAN_REVERTING  — ADX < 20, range-bound price action
    HIGH_VOL        — India VIX > 20, fear / wide intraday ranges
    LOW_VOL         — India VIX < 14, complacency / tight ranges
    CRISIS          — India VIX > 28, circuit-breaker territory → kill switch
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

log = logging.getLogger(__name__)


class Regime(str, Enum):
    TRENDING       = "TRENDING"
    MEAN_REVERTING = "MEAN_REVERTING"
    HIGH_VOL       = "HIGH_VOL"
    LOW_VOL        = "LOW_VOL"
    CRISIS         = "CRISIS"
    UNKNOWN        = "UNKNOWN"


# ── Thresholds (static — intentionally not dynamic to avoid circular dependency) ──
VIX_CRISIS         = 28.0
VIX_HIGH_VOL       = 20.0
VIX_LOW_VOL        = 14.0
ADX_TRENDING       = 25.0
ADX_MEAN_REVERTING = 20.0
BREADTH_BULLISH    = 1.5    # advance/decline ratio above this → healthy breadth
BREADTH_BEARISH    = 0.67   # advance/decline ratio below this → weak breadth


@dataclass
class RegimeResult:
    regime: Regime
    india_vix: float
    adx: float
    advance_decline_ratio: float
    nifty_change_pct: float
    confidence: float           # 0.0–1.0 — how clearly the regime fits
    notes: list[str] = field(default_factory=list)

    @property
    def is_crisis(self) -> bool:
        return self.regime == Regime.CRISIS

    @property
    def is_high_risk(self) -> bool:
        return self.regime in (Regime.CRISIS, Regime.HIGH_VOL)

    def __str__(self) -> str:
        return (
            f"Regime={self.regime.value} | VIX={self.india_vix:.1f} | "
            f"ADX={self.adx:.1f} | A/D={self.advance_decline_ratio:.2f} | "
            f"Nifty={self.nifty_change_pct:+.2f}% | conf={self.confidence:.2f}"
        )


class RegimeDetector:
    """
    Determines current market regime from a snapshot of market indicators.

    Priority order (highest overrides lower):
        1. CRISIS         — VIX >= 28  (always wins)
        2. HIGH_VOL       — VIX >= 20
        3. LOW_VOL        — VIX < 14
        4. TRENDING       — ADX > 25 (after VIX checks pass)
        5. MEAN_REVERTING — ADX < 20 (default range-bound)
        6. UNKNOWN        — insufficient data

    Mixed signals (e.g. VIX = 17, ADX = 27) → use confidence scoring
    to pick the dominant regime and note the conflict.
    """

    async def detect(self, snapshot: dict) -> str:
        """
        Async interface for tests and external callers.
        Accepts a dict snapshot with keys: vix, adx, atr_pct, bb_width_pct,
        trend_slope, volume_ratio, advance_decline_ratio.
        Returns the regime string (e.g. "CRISIS", "TRENDING").
        """
        result = self._detect_sync(
            india_vix=snapshot.get("vix", 16.0),
            adx=snapshot.get("adx", 20.0),
            advance_decline_ratio=snapshot.get("advance_decline_ratio", 1.0),
            nifty_change_pct=snapshot.get("trend_slope", 0.0),
        )
        return result.regime.value

    def _detect_sync(
        self,
        india_vix: float,
        adx: float,
        advance_decline_ratio: float,
        nifty_change_pct: float,
        nifty_5d_change_pct: Optional[float] = None,
        fii_net_flow_cr: Optional[float] = None,
    ) -> RegimeResult:
        """
        Classify current regime from indicator values.

        Args:
            india_vix:              India VIX index value (e.g. 16.5).
            adx:                    Average Directional Index of Nifty 50 (14-period).
            advance_decline_ratio:  NSE advance count / decline count.
            nifty_change_pct:       Nifty 50 intraday change %.
            nifty_5d_change_pct:    Optional 5-day Nifty change % (improves confidence).
            fii_net_flow_cr:        Optional net FII flow in ₹ crore (positive = buying).

        Returns:
            RegimeResult with regime label, confidence, and diagnostic notes.
        """
        notes: list[str] = []

        # ── 1. CRISIS — hard override ─────────────────────────────────────
        if india_vix >= VIX_CRISIS:
            notes.append(f"VIX={india_vix:.1f} >= {VIX_CRISIS} — CRISIS, kill switch active")
            return RegimeResult(
                regime=Regime.CRISIS,
                india_vix=india_vix,
                adx=adx,
                advance_decline_ratio=advance_decline_ratio,
                nifty_change_pct=nifty_change_pct,
                confidence=1.0,
                notes=notes,
            )

        # ── Compute a score per regime (higher = stronger fit) ────────────
        scores: dict[Regime, float] = {
            Regime.HIGH_VOL:       0.0,
            Regime.LOW_VOL:        0.0,
            Regime.TRENDING:       0.0,
            Regime.MEAN_REVERTING: 0.0,
        }

        # ── 2. HIGH_VOL signals ───────────────────────────────────────────
        if india_vix >= VIX_HIGH_VOL:
            scores[Regime.HIGH_VOL] += 2.0
            notes.append(f"VIX={india_vix:.1f} >= {VIX_HIGH_VOL}")
        if abs(nifty_change_pct) >= 1.5:
            scores[Regime.HIGH_VOL] += 1.0
            notes.append(f"Large intraday move {nifty_change_pct:+.2f}%")
        if advance_decline_ratio < BREADTH_BEARISH:
            scores[Regime.HIGH_VOL] += 0.5
            notes.append(f"Weak breadth A/D={advance_decline_ratio:.2f}")
        if fii_net_flow_cr is not None and fii_net_flow_cr < -2000:
            scores[Regime.HIGH_VOL] += 0.5
            notes.append(f"Heavy FII selling ₹{fii_net_flow_cr:.0f}cr")

        # ── 3. LOW_VOL signals ────────────────────────────────────────────
        if india_vix < VIX_LOW_VOL:
            scores[Regime.LOW_VOL] += 2.0
            notes.append(f"VIX={india_vix:.1f} < {VIX_LOW_VOL}")
        if abs(nifty_change_pct) <= 0.3:
            scores[Regime.LOW_VOL] += 0.5
            notes.append(f"Tight intraday range {nifty_change_pct:+.2f}%")

        # ── 4. TRENDING signals ───────────────────────────────────────────
        if adx >= ADX_TRENDING:
            scores[Regime.TRENDING] += 2.0
            notes.append(f"ADX={adx:.1f} >= {ADX_TRENDING}")
        if nifty_5d_change_pct is not None and abs(nifty_5d_change_pct) >= 2.0:
            scores[Regime.TRENDING] += 1.0
            notes.append(f"5d momentum {nifty_5d_change_pct:+.2f}%")
        if advance_decline_ratio >= BREADTH_BULLISH:
            scores[Regime.TRENDING] += 0.5
            notes.append(f"Strong breadth A/D={advance_decline_ratio:.2f}")
        if fii_net_flow_cr is not None and fii_net_flow_cr > 2000:
            scores[Regime.TRENDING] += 0.5
            notes.append(f"Heavy FII buying ₹{fii_net_flow_cr:.0f}cr")

        # ── 5. MEAN_REVERTING signals ─────────────────────────────────────
        if adx <= ADX_MEAN_REVERTING:
            scores[Regime.MEAN_REVERTING] += 2.0
            notes.append(f"ADX={adx:.1f} <= {ADX_MEAN_REVERTING}")
        if VIX_LOW_VOL <= india_vix <= VIX_HIGH_VOL:
            scores[Regime.MEAN_REVERTING] += 0.5
            notes.append(f"VIX in neutral zone ({VIX_LOW_VOL}–{VIX_HIGH_VOL})")

        # ── Pick winner ───────────────────────────────────────────────────
        best_regime = max(scores, key=lambda r: scores[r])
        best_score  = scores[best_regime]
        total_score = sum(scores.values()) or 1.0

        if best_score == 0:
            log.warning("RegimeDetector: all scores are 0 — insufficient data")
            return RegimeResult(
                regime=Regime.UNKNOWN,
                india_vix=india_vix,
                adx=adx,
                advance_decline_ratio=advance_decline_ratio,
                nifty_change_pct=nifty_change_pct,
                confidence=0.0,
                notes=["Insufficient data to determine regime"],
            )

        confidence = best_score / total_score

        result = RegimeResult(
            regime=best_regime,
            india_vix=india_vix,
            adx=adx,
            advance_decline_ratio=advance_decline_ratio,
            nifty_change_pct=nifty_change_pct,
            confidence=confidence,
            notes=notes,
        )
        log.debug("RegimeDetector: %s", result)
        return result

    def detect_from_state(self, market_state: dict) -> RegimeResult:
        """
        Convenience wrapper that accepts the market_state dict gathered by
        WeightCalibrationAgent._gather_market_state().
        """
        return self._detect_sync(
            india_vix=market_state.get("india_vix", 16.0),
            adx=market_state.get("adx", 20.0),
            advance_decline_ratio=market_state.get("market_breadth", 1.0),
            nifty_change_pct=market_state.get("nifty_change_pct", 0.0),
            nifty_5d_change_pct=market_state.get("nifty_5d_change_pct"),
            fii_net_flow_cr=market_state.get("fii_dii_flow", {}).get("fii_net_cr"),
        )
