"""
NEXUS-II — Sentiment Analyst (Tier 1)

Two-stage analysis:
  Stage 1 (Claude Haiku — fast NLP):
      Entity extraction from raw headlines → which companies / events are mentioned
  Stage 2 (Claude Sonnet — synthesis):
      Combines FinBERT scores, entity list, mention velocity, and social narratives
      into a directional sentiment assessment for the target symbol.

Sentiment signals that trigger sub-agents:
  - Score > +0.60 sustained 2+ hours → feeds SentimentAgent BUY signal
  - Score < -0.60 sustained 2+ hours → feeds SentimentAgent SELL signal
  - Swing > 0.40 within 30 min → immediate escalation signal

Output: AnalystReport with BUY / SELL / HOLD bias and 0–1 conviction.
"""
from __future__ import annotations

import json
import logging

from core.agents.analysts import AnalystReport, BaseAnalyst
from core.agents.base_agent import Action

log = logging.getLogger(__name__)

# ── System prompts ─────────────────────────────────────────────────────────────

_ENTITY_EXTRACTION_SYSTEM = """You are a financial entity extractor. Given a list of news headlines about Indian markets, extract a JSON list of named entities relevant to a specific stock ticker.

Return ONLY a JSON array of strings (entity names). Example: ["TCS", "Infosys", "RBI rate hike", "Q3 earnings beat"]

Include: company names, indices, macro events, regulatory actions, earnings events.
Exclude: generic market commentary with no named subject."""

_SENTIMENT_SYSTEM = """You are the Sentiment Analyst for NEXUS-II, an AI trading system for Indian markets (NSE/BSE).

Your job: Given FinBERT sentiment scores, entity mentions, velocity, and social narratives for a specific stock, produce a structured sentiment assessment.

== SENTIMENT THRESHOLDS ==
- Score > +0.60 sustained 2+ hours → strong BUY signal
- Score < -0.60 sustained 2+ hours → strong SELL signal
- Rapid swing > 0.40 within 30 minutes → immediate escalation (raise conviction)
- Scores between -0.60 and +0.60 → HOLD unless velocity is extreme

== CONVICTION MODIFIERS ==
- High mention velocity (> 5× baseline): +0.15 conviction
- Negative sentiment with falling price confirmation: +0.10
- Positive sentiment with insider/FII buying confirmation: +0.10
- Conflicting signals (bullish headlines + bearish price): −0.15
- Single source only (not corroborated): −0.10

== OUTPUT FORMAT ==
Return ONLY valid JSON (no markdown):
{
  "signal": "BUY" | "SELL" | "HOLD",
  "conviction": <float 0.0–1.0>,
  "summary": "<2–3 sentences synthesising the sentiment picture>",
  "key_findings": ["<finding 1>", "<finding 2>"],
  "dominant_narrative": "<one sentence — the prevailing market story>",
  "entities_relevant": ["<entity 1>", ...],
  "swing_alert": <bool — true if score swung > 0.40 in last 30 min>
}
"""

# ── SentimentAnalyst ──────────────────────────────────────────────────────────

class SentimentAnalyst(BaseAnalyst):
    """
    Tier-1 Sentiment Analyst.

    Expects market_data to contain a "sentiment" sub-dict with:
        score:              float (-1.0 to +1.0) — composite FinBERT score
        score_1h_ago:       float — score 1 hour ago (for velocity / swing calc)
        score_30m_ago:      float — score 30 min ago (for swing detection)
        mentions:           int   — total mentions in last 2 hours
        mentions_baseline:  int   — typical mentions per 2 hours
        velocity:           float — mentions per hour (last 30 min rate)
        finbert_scores:     list[float] — per-headline FinBERT scores
        headlines:          list[str]   — raw headline text (last 20)
        social_narratives:  list[str]   — curated social media narratives
        sustained_hours:    float       — hours score has stayed above/below threshold
    """

    ANALYST_NAME = "sentiment"

    async def analyze(self, symbol: str, market_data: dict) -> AnalystReport:
        sent = market_data.get("sentiment", {})

        headlines: list[str] = sent.get("headlines", [])
        score: float = float(sent.get("score", 0.0))
        score_30m: float = float(sent.get("score_30m_ago", 0.0))
        score_1h: float = float(sent.get("score_1h_ago", 0.0))
        mentions: int = int(sent.get("mentions", 0))
        baseline: int = int(sent.get("mentions_baseline", 1))
        velocity: float = float(sent.get("velocity", 0.0))
        finbert: list = sent.get("finbert_scores", [])
        narratives: list = sent.get("social_narratives", [])
        sustained_hours: float = float(sent.get("sustained_hours", 0.0))

        raw_data = {
            "score": score,
            "score_30m_ago": score_30m,
            "score_1h_ago": score_1h,
            "swing_30m": round(score - score_30m, 4),
            "swing_1h": round(score - score_1h, 4),
            "mentions": mentions,
            "mentions_baseline": baseline,
            "velocity_multiplier": round(velocity / max(baseline / 2, 1), 2),
            "finbert_avg": round(sum(finbert) / len(finbert), 4) if finbert else 0.0,
            "finbert_scores": finbert[:10],   # cap at 10 for prompt size
            "headlines": headlines[:15],
            "social_narratives": narratives[:5],
            "sustained_hours": sustained_hours,
        }

        # ── Stage 1: entity extraction (Haiku — fast) ─────────────────────────
        entities: list[str] = []
        if headlines:
            try:
                entity_user = (
                    f"Stock ticker: {symbol}\n\nHeadlines:\n"
                    + "\n".join(f"- {h}" for h in headlines[:15])
                )
                entity_text = await self._call_claude_fast(
                    _ENTITY_EXTRACTION_SYSTEM, entity_user
                )
                entities = json.loads(self._strip_fences(entity_text))
                if not isinstance(entities, list):
                    entities = []
            except Exception as exc:
                log.warning("SentimentAnalyst: entity extraction failed: %s", exc)

        # ── Stage 2: synthesis (Sonnet) ────────────────────────────────────────
        raw_data["entities_extracted"] = entities

        user_content = (
            f"Analyse sentiment for **{symbol}**.\n\n"
            f"```json\n{json.dumps(raw_data, indent=2, default=str)}\n```"
        )

        try:
            raw_text = await self._call_claude(_SENTIMENT_SYSTEM, user_content)
            parsed = json.loads(self._strip_fences(raw_text))

            signal = Action(parsed.get("signal", "HOLD").upper())
            conviction = float(parsed.get("conviction", 0.0))
            summary = parsed.get("summary", "")
            key_findings = parsed.get("key_findings", [])
            dominant_narrative = parsed.get("dominant_narrative", "")
            entities_relevant = parsed.get("entities_relevant", entities)
            swing_alert = bool(parsed.get("swing_alert", False))

        except Exception as exc:
            return self._fallback_report(symbol, raw_data, str(exc))

        return AnalystReport(
            analyst=self.ANALYST_NAME,
            symbol=symbol,
            signal=signal,
            conviction=conviction,
            summary=summary,
            key_findings=key_findings,
            raw_data=raw_data,
            metadata={
                "dominant_narrative": dominant_narrative,
                "entities_relevant": entities_relevant,
                "swing_alert": swing_alert,
            },
        )
