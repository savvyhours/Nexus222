"""
NEXUS-II — Sentiment Store
Persists and retrieves sentiment readings from Supabase (sentiment_cache table).

Schema (from design doc):
    CREATE TABLE sentiment_cache (
        id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        symbol      TEXT NOT NULL,
        score       DECIMAL,
        source      TEXT,
        tier        INT,
        created_at  TIMESTAMPTZ DEFAULT NOW()
    );

Tiers:
    1 = FinBERT only
    2 = Haiku-enriched
    3 = Combined (FinBERT + Haiku blend)

The store writes every new reading and exposes:
  • get_latest(symbol)         → most recent score for a symbol
  • get_window(symbol, hours)  → readings from the last N hours
  • get_sector_snapshot()      → latest score per symbol for the aggregator

Usage:
    store = SentimentStore()
    await store.save("INFY", score=0.72, source="combined", tier=3)
    latest = await store.get_latest("INFY")
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")

TABLE = "sentiment_cache"


# ── Row dataclass ─────────────────────────────────────────────────────────────

@dataclass
class SentimentRow:
    symbol: str
    score: float
    source: str
    tier: int
    created_at: datetime

    @classmethod
    def from_supabase(cls, row: dict) -> "SentimentRow":
        ts_raw = row.get("created_at", "")
        try:
            ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            ts = datetime.now(timezone.utc)
        return cls(
            symbol=row.get("symbol", ""),
            score=float(row.get("score", 0.0)),
            source=row.get("source", "unknown"),
            tier=int(row.get("tier", 1)),
            created_at=ts,
        )


# ── Sentiment Store ───────────────────────────────────────────────────────────

class SentimentStore:
    """
    Supabase-backed persistence layer for sentiment readings.

    Falls back gracefully to in-memory storage when Supabase is unavailable
    so the rest of the system keeps running (paper-trade mode, unit tests).

    Parameters
    ----------
    supabase_url      : Supabase project URL. Reads SUPABASE_URL env var if omitted.
    supabase_key      : Supabase service role key. Reads SUPABASE_SERVICE_ROLE_KEY.
    fallback_in_memory: If True (default), cache writes in-memory when DB is down.
    """

    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        fallback_in_memory: bool = True,
    ) -> None:
        self._url = supabase_url or os.getenv("SUPABASE_URL")
        self._key = supabase_key or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        self._fallback = fallback_in_memory
        self._client = None
        self._memory: List[dict] = []  # in-memory fallback buffer

    # ── Client ────────────────────────────────────────────────────────────

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not self._url or not self._key:
            log.warning(
                "Supabase credentials missing — SentimentStore running in-memory only."
            )
            return None
        try:
            from supabase import create_client  # type: ignore
            self._client = create_client(self._url, self._key)
            return self._client
        except ImportError:
            log.warning("supabase-py not installed — SentimentStore in-memory only.")
            return None
        except Exception as exc:
            log.error("Failed to create Supabase client: %s", exc)
            return None

    # ── Write ─────────────────────────────────────────────────────────────

    async def save(
        self,
        symbol: str,
        score: float,
        source: str = "combined",
        tier: int = 3,
    ) -> bool:
        """
        Persist a sentiment reading.

        Parameters
        ----------
        symbol : NSE/BSE ticker.
        score  : Normalized sentiment score [-1, +1].
        source : "finbert" | "haiku" | "combined".
        tier   : 1 = FinBERT, 2 = Haiku, 3 = Combined.

        Returns
        -------
        True on successful Supabase write; False on failure (falls back to memory).
        """
        row = {
            "symbol": symbol.upper(),
            "score": round(float(score), 6),
            "source": source,
            "tier": tier,
        }
        client = self._get_client()
        if client is None:
            if self._fallback:
                row["created_at"] = datetime.now(timezone.utc).isoformat()
                self._memory.append(row)
            return False

        try:
            import asyncio
            await asyncio.to_thread(
                lambda: client.table(TABLE).insert(row).execute()
            )
            return True
        except Exception as exc:
            log.error("Supabase insert failed for %s: %s", symbol, exc)
            if self._fallback:
                row["created_at"] = datetime.now(timezone.utc).isoformat()
                self._memory.append(row)
            return False

    async def save_batch(self, rows: List[dict]) -> bool:
        """
        Bulk-insert multiple readings in a single Supabase call.

        Each dict must have: symbol, score, source, tier.
        """
        if not rows:
            return True
        normalized = [
            {
                "symbol": r["symbol"].upper(),
                "score": round(float(r["score"]), 6),
                "source": r.get("source", "combined"),
                "tier": r.get("tier", 3),
            }
            for r in rows
        ]
        client = self._get_client()
        if client is None:
            if self._fallback:
                ts = datetime.now(timezone.utc).isoformat()
                for r in normalized:
                    r["created_at"] = ts
                self._memory.extend(normalized)
            return False

        try:
            import asyncio
            await asyncio.to_thread(
                lambda: client.table(TABLE).insert(normalized).execute()
            )
            return True
        except Exception as exc:
            log.error("Supabase batch insert failed: %s", exc)
            if self._fallback:
                ts = datetime.now(timezone.utc).isoformat()
                for r in normalized:
                    r["created_at"] = ts
                self._memory.extend(normalized)
            return False

    # ── Read ──────────────────────────────────────────────────────────────

    async def get_latest(self, symbol: str) -> Optional[SentimentRow]:
        """Return the most recent sentiment reading for a symbol."""
        symbol = symbol.upper()
        client = self._get_client()

        if client is None:
            rows = [r for r in self._memory if r["symbol"] == symbol]
            if not rows:
                return None
            row = max(rows, key=lambda r: r["created_at"])
            return SentimentRow.from_supabase(row)

        try:
            import asyncio
            result = await asyncio.to_thread(
                lambda: (
                    client.table(TABLE)
                    .select("*")
                    .eq("symbol", symbol)
                    .order("created_at", desc=True)
                    .limit(1)
                    .execute()
                )
            )
            data = result.data
            if not data:
                return None
            return SentimentRow.from_supabase(data[0])
        except Exception as exc:
            log.error("Supabase get_latest failed for %s: %s", symbol, exc)
            return None

    async def get_window(
        self, symbol: str, hours: int = 4
    ) -> List[SentimentRow]:
        """
        Return all readings for a symbol within the last N hours.

        Ordered oldest → newest.
        """
        symbol = symbol.upper()
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        client = self._get_client()

        if client is None:
            rows = [
                r for r in self._memory
                if r["symbol"] == symbol
                and r.get("created_at", "") >= cutoff.isoformat()
            ]
            return [SentimentRow.from_supabase(r) for r in rows]

        try:
            import asyncio
            result = await asyncio.to_thread(
                lambda: (
                    client.table(TABLE)
                    .select("*")
                    .eq("symbol", symbol)
                    .gte("created_at", cutoff.isoformat())
                    .order("created_at", desc=False)
                    .execute()
                )
            )
            return [SentimentRow.from_supabase(r) for r in result.data]
        except Exception as exc:
            log.error("Supabase get_window failed for %s: %s", symbol, exc)
            return []

    async def get_sector_snapshot(self, hours: int = 1) -> Dict[str, float]:
        """
        Return the latest score per symbol for all symbols written in the last N hours.

        Used by SectorAggregator to re-hydrate from DB on startup.
        Returns dict: {symbol → score}
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        client = self._get_client()

        if client is None:
            rows = [r for r in self._memory if r.get("created_at", "") >= cutoff.isoformat()]
            snapshot: Dict[str, float] = {}
            for r in rows:
                snapshot[r["symbol"]] = float(r["score"])
            return snapshot

        try:
            import asyncio
            result = await asyncio.to_thread(
                lambda: (
                    client.table(TABLE)
                    .select("symbol, score, created_at")
                    .gte("created_at", cutoff.isoformat())
                    .order("created_at", desc=True)
                    .execute()
                )
            )
            snapshot: Dict[str, float] = {}
            for row in result.data:
                sym = row["symbol"]
                if sym not in snapshot:  # keep most recent (already sorted desc)
                    snapshot[sym] = float(row["score"])
            return snapshot
        except Exception as exc:
            log.error("Supabase get_sector_snapshot failed: %s", exc)
            return {}

    # ── Maintenance ───────────────────────────────────────────────────────

    async def purge_old(self, days: int = 7) -> int:
        """
        Delete readings older than N days. Returns number of rows deleted.
        Only runs against Supabase (in-memory buffer is not purged here).
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        client = self._get_client()
        if client is None:
            return 0
        try:
            import asyncio
            result = await asyncio.to_thread(
                lambda: (
                    client.table(TABLE)
                    .delete()
                    .lt("created_at", cutoff.isoformat())
                    .execute()
                )
            )
            count = len(result.data) if result.data else 0
            log.info("Purged %d old sentiment rows (older than %d days).", count, days)
            return count
        except Exception as exc:
            log.error("Supabase purge_old failed: %s", exc)
            return 0
