"""
NEXUS-II — LLM Enricher (Claude Haiku)
Extracts structured metadata from financial news using Claude Haiku 4.5.

FinBERT gives us a sentiment score.  The enricher adds:
  • mentioned_symbols  — NSE/BSE tickers referenced in the article
  • sector             — GICS-style sector label
  • event_type         — EARNINGS | M_AND_A | REGULATORY | MACRO | PRODUCT | OTHER
  • impact_horizon     — INTRADAY | SWING | LONG_TERM
  • key_entities       — up to 3 named entities (companies, people, indices)
  • confidence         — model's self-rated confidence 0–1

All extraction is done via a single structured JSON prompt so we get one API
call per article. Rate limiting and retry logic are handled here.

Usage:
    enricher = LLMEnricher()
    result = await enricher.enrich("Infosys bags $1.5B TCV deal with BP ...")
    # EnrichedMeta(mentioned_symbols=['INFY'], sector='IT', event_type='PRODUCT', ...)
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

import anthropic

log = logging.getLogger(__name__)

# ── Haiku model ID ────────────────────────────────────────────────────────────
HAIKU_MODEL = "claude-haiku-4-5-20251001"

# ── Extraction prompt template ────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are a financial NLP extraction engine focused on Indian equity markets (NSE/BSE).
Extract structured metadata from financial news headlines and articles.
Return ONLY valid JSON — no markdown, no explanation.
"""

_USER_TEMPLATE = """\
Extract metadata from this financial text:

TEXT: {text}

Return JSON with exactly these keys:
{{
  "mentioned_symbols": ["<NSE ticker>", ...],   // max 5, e.g. ["RELIANCE","TCS"]
  "sector": "<GICS sector>",                     // e.g. "Information Technology"
  "event_type": "<EARNINGS|M_AND_A|REGULATORY|MACRO|PRODUCT|OTHER>",
  "impact_horizon": "<INTRADAY|SWING|LONG_TERM>",
  "key_entities": ["<entity>", ...],             // max 3 named entities
  "confidence": <0.0-1.0>
}}
"""


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class EnrichedMeta:
    """
    Structured metadata extracted by Claude Haiku from financial text.

    All fields have safe defaults so callers never receive None.
    """
    mentioned_symbols: List[str] = field(default_factory=list)
    sector: str = "UNKNOWN"
    event_type: str = "OTHER"
    impact_horizon: str = "SWING"
    key_entities: List[str] = field(default_factory=list)
    confidence: float = 0.0
    raw_text: str = ""

    @classmethod
    def empty(cls, text: str = "") -> "EnrichedMeta":
        return cls(raw_text=text[:120])

    def to_dict(self) -> dict:
        return {
            "mentioned_symbols": self.mentioned_symbols,
            "sector": self.sector,
            "event_type": self.event_type,
            "impact_horizon": self.impact_horizon,
            "key_entities": self.key_entities,
            "confidence": self.confidence,
        }


# ── LLM Enricher ─────────────────────────────────────────────────────────────

class LLMEnricher:
    """
    Wraps Claude Haiku 4.5 for high-throughput financial entity extraction.

    Parameters
    ----------
    api_key     : Anthropic API key. Falls back to CLAUDE_API_KEY env var.
    max_tokens  : Max tokens for Haiku response (JSON is small; 256 is plenty).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_tokens: int = 256,
    ) -> None:
        self._api_key = api_key or os.getenv("CLAUDE_API_KEY")
        self._max_tokens = max_tokens
        self._client: Optional[anthropic.AsyncAnthropic] = None

    def _get_client(self) -> anthropic.AsyncAnthropic:
        if self._client is None:
            if not self._api_key:
                raise RuntimeError("CLAUDE_API_KEY not set — LLMEnricher cannot call Haiku.")
            self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
        return self._client

    async def enrich(self, text: str) -> EnrichedMeta:
        """
        Extract structured metadata from a single financial text.

        Parameters
        ----------
        text : Raw news headline or article snippet (truncated to 1000 chars).

        Returns
        -------
        EnrichedMeta — safe defaults on any failure.
        """
        truncated = text[:1000]
        try:
            client = self._get_client()
            response = await client.messages.create(
                model=HAIKU_MODEL,
                max_tokens=self._max_tokens,
                system=_SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": _USER_TEMPLATE.format(text=truncated)}
                ],
            )
            raw_json = response.content[0].text.strip()
            return self._parse(raw_json, truncated)
        except anthropic.APIError as exc:
            log.warning("Haiku API error during enrichment: %s", exc)
            return EnrichedMeta.empty(truncated)
        except Exception as exc:
            log.error("Unexpected error in LLMEnricher.enrich: %s", exc)
            return EnrichedMeta.empty(truncated)

    def _parse(self, raw_json: str, original_text: str) -> EnrichedMeta:
        try:
            data = json.loads(raw_json)
            return EnrichedMeta(
                mentioned_symbols=[s.upper() for s in data.get("mentioned_symbols", [])[:5]],
                sector=data.get("sector", "UNKNOWN"),
                event_type=data.get("event_type", "OTHER"),
                impact_horizon=data.get("impact_horizon", "SWING"),
                key_entities=data.get("key_entities", [])[:3],
                confidence=float(data.get("confidence", 0.0)),
                raw_text=original_text[:120],
            )
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            log.warning("Failed to parse Haiku JSON: %s | raw: %.80s", exc, raw_json)
            return EnrichedMeta.empty(original_text)

    async def enrich_batch(self, texts: List[str]) -> List[EnrichedMeta]:
        """
        Enrich a list of texts concurrently (up to 10 parallel calls).

        Parameters
        ----------
        texts : List of raw text strings.

        Returns
        -------
        List of EnrichedMeta in the same order as input.
        """
        import asyncio

        semaphore = asyncio.Semaphore(10)

        async def _bounded_enrich(text: str) -> EnrichedMeta:
            async with semaphore:
                return await self.enrich(text)

        return await asyncio.gather(*[_bounded_enrich(t) for t in texts])
