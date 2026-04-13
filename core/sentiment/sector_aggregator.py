"""
NEXUS-II — Sector Aggregator
Aggregates per-symbol sentiment scores into sector-level and index-level views.

Flow:
  1. Caller pushes SentimentResult + EnrichedMeta items via `push()`.
  2. `get_sector_score(sector)` returns a weighted average over a rolling window.
  3. `get_index_score(index)` returns a composite for NSE indices (NIFTY50, NIFTYIT, etc.)
  4. `get_top_movers()` returns the symbols with the largest absolute sentiment shift.

Scores are kept in-memory; SentimentStore is responsible for Supabase persistence.
The aggregator is designed to be called by the SentimentAgent every market cycle.

Usage:
    agg = SectorAggregator()
    agg.push("INFY", "Information Technology", score=0.72)
    agg.push("TCS",  "Information Technology", score=0.55)
    sector_score = agg.get_sector_score("Information Technology")  # 0.635
    index_score  = agg.get_index_score("NIFTYIT")                  # 0.635
"""
from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")

# ── Index → constituent sectors / symbols mapping ─────────────────────────────
# Used to compute index-level sentiment from underlying sector scores.

INDEX_SECTOR_MAP: Dict[str, List[str]] = {
    "NIFTY50": [
        "Information Technology", "Financial Services", "Oil Gas & Consumable Fuels",
        "Fast Moving Consumer Goods", "Automobile", "Healthcare",
        "Metals & Mining", "Telecommunication", "Power", "Construction",
    ],
    "NIFTYIT": ["Information Technology"],
    "NIFTYPHARMA": ["Healthcare"],
    "NIFTYBANK": ["Financial Services"],
    "BSE": [
        "Information Technology", "Financial Services", "Oil Gas & Consumable Fuels",
        "Fast Moving Consumer Goods", "Automobile", "Healthcare",
        "Metals & Mining", "Telecommunication", "Power", "Construction",
        "Capital Goods", "Chemicals", "Consumer Durables", "Realty",
    ],
    "NIFTY500": [],  # Empty = all sectors included
}

# How many recent readings to keep per symbol (rolling window).
WINDOW_SIZE = 50


# ── Per-symbol reading ────────────────────────────────────────────────────────

@dataclass
class SymbolReading:
    symbol: str
    sector: str
    score: float          # normalized [-1, +1]
    confidence: float     # 0–1 from FinBERT raw_score or Haiku confidence
    source: str           # "finbert" | "haiku" | "combined"
    ts: datetime = field(default_factory=lambda: datetime.now(IST))


# ── Aggregated result ─────────────────────────────────────────────────────────

@dataclass
class AggregatedScore:
    name: str              # sector name or index name
    score: float           # weighted avg in [-1, +1]
    symbol_count: int      # distinct symbols contributing
    reading_count: int     # total readings averaged
    bullish_pct: float     # fraction of readings with score > 0.1
    bearish_pct: float     # fraction of readings with score < -0.1
    ts: datetime = field(default_factory=lambda: datetime.now(IST))

    @property
    def label(self) -> str:
        if self.score > 0.15:
            return "BULLISH"
        if self.score < -0.15:
            return "BEARISH"
        return "NEUTRAL"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "score": round(self.score, 4),
            "label": self.label,
            "symbol_count": self.symbol_count,
            "reading_count": self.reading_count,
            "bullish_pct": round(self.bullish_pct, 3),
            "bearish_pct": round(self.bearish_pct, 3),
            "ts": self.ts.isoformat(),
        }


# ── Sector Aggregator ─────────────────────────────────────────────────────────

class SectorAggregator:
    """
    In-memory rolling aggregator for sector and index sentiment.

    Thread-safety note: not thread-safe. Use from a single asyncio event loop.
    """

    def __init__(self, window_size: int = WINDOW_SIZE) -> None:
        self._window_size = window_size
        # symbol → deque of SymbolReading
        self._symbol_readings: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self._window_size)
        )
        # symbol → latest sector (updated on every push)
        self._symbol_sector: Dict[str, str] = {}

    # ── Ingestion ─────────────────────────────────────────────────────────

    def push(
        self,
        symbol: str,
        sector: str,
        score: float,
        confidence: float = 1.0,
        source: str = "combined",
    ) -> None:
        """
        Record a new sentiment reading for a symbol.

        Parameters
        ----------
        symbol     : NSE/BSE ticker (e.g. "RELIANCE").
        sector     : GICS-style sector string from EnrichedMeta.
        score      : Normalized sentiment score in [-1, +1].
        confidence : Confidence weight for this reading (default 1.0).
        source     : Data origin label ("finbert" | "haiku" | "combined").
        """
        symbol = symbol.upper()
        reading = SymbolReading(
            symbol=symbol,
            sector=sector,
            score=max(-1.0, min(1.0, score)),
            confidence=max(0.0, min(1.0, confidence)),
            source=source,
        )
        self._symbol_readings[symbol].append(reading)
        self._symbol_sector[symbol] = sector

    def push_result(
        self,
        symbol: str,
        sector: str,
        finbert_score: float,
        haiku_confidence: float = 0.5,
    ) -> None:
        """
        Convenience method: combine FinBERT + Haiku signals into one reading.

        The combined score is a weighted blend:
            0.7 × finbert_score + 0.3 × (haiku_confidence projected to [-1,+1])
        The haiku_confidence field (0–1) is treated as directional only when
        finbert already has a direction; it modulates magnitude.
        """
        blended = 0.7 * finbert_score + 0.3 * (finbert_score * haiku_confidence)
        self.push(symbol, sector, score=blended, confidence=haiku_confidence, source="combined")

    # ── Sector-level aggregation ──────────────────────────────────────────

    def get_sector_score(self, sector: str) -> Optional[AggregatedScore]:
        """
        Compute a confidence-weighted average sentiment for a sector.

        Returns None if no symbols in the sector have been pushed.
        """
        symbols = [s for s, sec in self._symbol_sector.items() if sec == sector]
        if not symbols:
            return None
        return self._aggregate(sector, symbols)

    def get_all_sector_scores(self) -> List[AggregatedScore]:
        """Return aggregated scores for every sector that has readings."""
        sectors = set(self._symbol_sector.values())
        results = []
        for sector in sorted(sectors):
            score = self.get_sector_score(sector)
            if score:
                results.append(score)
        return results

    # ── Index-level aggregation ───────────────────────────────────────────

    def get_index_score(self, index: str) -> Optional[AggregatedScore]:
        """
        Compute sentiment for an NSE/BSE index by averaging its constituent sectors.

        For NIFTY500 (empty sector list) all known sectors are included.
        Returns None if no relevant readings exist.
        """
        index = index.upper()
        target_sectors = INDEX_SECTOR_MAP.get(index, [])
        if target_sectors:
            symbols = [s for s, sec in self._symbol_sector.items() if sec in target_sectors]
        else:
            # NIFTY500 or unknown → all symbols
            symbols = list(self._symbol_sector.keys())

        if not symbols:
            return None
        return self._aggregate(index, symbols)

    # ── Top movers ────────────────────────────────────────────────────────

    def get_top_movers(self, n: int = 10) -> List[Tuple[str, float]]:
        """
        Return the n symbols with the highest absolute latest sentiment score.

        Returns list of (symbol, score) sorted by abs(score) descending.
        """
        latest: List[Tuple[str, float]] = []
        for symbol, readings in self._symbol_readings.items():
            if readings:
                latest.append((symbol, readings[-1].score))
        latest.sort(key=lambda x: abs(x[1]), reverse=True)
        return latest[:n]

    def get_symbol_score(self, symbol: str) -> Optional[float]:
        """Return the latest sentiment score for a specific symbol, or None."""
        readings = self._symbol_readings.get(symbol.upper())
        if not readings:
            return None
        return readings[-1].score

    # ── Internals ─────────────────────────────────────────────────────────

    def _aggregate(self, name: str, symbols: List[str]) -> AggregatedScore:
        all_readings: List[SymbolReading] = []
        for sym in symbols:
            all_readings.extend(self._symbol_readings.get(sym, []))

        if not all_readings:
            return AggregatedScore(
                name=name, score=0.0, symbol_count=0,
                reading_count=0, bullish_pct=0.0, bearish_pct=0.0,
            )

        total_weight = sum(r.confidence for r in all_readings)
        if total_weight == 0:
            weighted_avg = 0.0
        else:
            weighted_avg = sum(r.score * r.confidence for r in all_readings) / total_weight

        bullish = sum(1 for r in all_readings if r.score > 0.1)
        bearish = sum(1 for r in all_readings if r.score < -0.1)
        n = len(all_readings)

        return AggregatedScore(
            name=name,
            score=round(weighted_avg, 4),
            symbol_count=len(symbols),
            reading_count=n,
            bullish_pct=round(bullish / n, 3),
            bearish_pct=round(bearish / n, 3),
        )

    def clear(self) -> None:
        """Reset all in-memory state (called between sessions or on crisis reset)."""
        self._symbol_readings.clear()
        self._symbol_sector.clear()
        log.info("SectorAggregator state cleared.")
