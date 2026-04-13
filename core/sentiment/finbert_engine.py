"""
NEXUS-II — FinBERT Engine
Runs ProsusAI/finbert to classify financial text as POSITIVE / NEGATIVE / NEUTRAL
and returns a normalized sentiment score in [-1, +1].

The model is loaded once at process startup (lazy singleton). Inference is
synchronous but wrapped in asyncio.to_thread() so it does not block the event loop.

Usage:
    engine = FinBERTEngine()
    await engine.load()
    result = await engine.score("Infosys beats Q4 estimates; margin expands 200bps")
    # SentimentResult(label='POSITIVE', score=0.92, normalized=0.92)
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import List, Optional

log = logging.getLogger(__name__)

# Lazy imports so the module loads even when transformers is not installed.
_transformers_available: Optional[bool] = None


def _check_transformers() -> bool:
    global _transformers_available
    if _transformers_available is None:
        try:
            import transformers  # noqa: F401
            _transformers_available = True
        except ImportError:
            _transformers_available = False
    return _transformers_available


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class SentimentResult:
    """
    Single-document sentiment result from FinBERT.

    Attributes
    ----------
    label       : 'POSITIVE', 'NEGATIVE', or 'NEUTRAL'
    raw_score   : model confidence for the winning label (0–1)
    normalized  : signed score in [-1, +1].
                  POSITIVE → +raw_score, NEGATIVE → -raw_score, NEUTRAL → 0
    text        : truncated source text (first 120 chars)
    """
    label: str
    raw_score: float
    normalized: float
    text: str = ""

    @classmethod
    def neutral(cls, text: str = "") -> "SentimentResult":
        return cls(label="NEUTRAL", raw_score=0.0, normalized=0.0, text=text)


# ── FinBERT singleton ─────────────────────────────────────────────────────────

class FinBERTEngine:
    """
    Singleton wrapper around ProsusAI/finbert.

    The pipeline is loaded once and reused for all subsequent calls.
    If transformers is not installed, `score()` always returns NEUTRAL
    and logs a warning — the rest of the system continues to function.
    """

    _instance: Optional["FinBERTEngine"] = None
    MODEL_ID = "ProsusAI/finbert"
    MAX_LENGTH = 512  # FinBERT's context window

    def __new__(cls) -> "FinBERTEngine":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._pipeline = None
            cls._instance._loaded = False
        return cls._instance

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def load(self) -> None:
        """Load FinBERT in a thread pool (avoids blocking the event loop)."""
        if self._loaded:
            return
        if not _check_transformers():
            log.warning(
                "transformers not installed — FinBERT disabled. "
                "Install: pip install transformers torch"
            )
            return
        log.info("Loading FinBERT model: %s …", self.MODEL_ID)
        await asyncio.to_thread(self._load_sync)

    def _load_sync(self) -> None:
        from transformers import pipeline  # type: ignore
        self._pipeline = pipeline(
            task="text-classification",
            model=self.MODEL_ID,
            tokenizer=self.MODEL_ID,
            truncation=True,
            max_length=self.MAX_LENGTH,
            device=-1,  # CPU; set to 0 for CUDA
        )
        self._loaded = True
        log.info("FinBERT loaded successfully.")

    # ── Inference ─────────────────────────────────────────────────────────

    async def score(self, text: str) -> SentimentResult:
        """
        Score a single piece of financial text.

        Parameters
        ----------
        text : Raw news headline, article snippet, or tweet.

        Returns
        -------
        SentimentResult with label, raw_score, and normalized score.
        """
        if not self._loaded:
            log.debug("FinBERT not loaded; returning NEUTRAL for: %.60s", text)
            return SentimentResult.neutral(text[:120])

        snippet = text[:120]
        try:
            result = await asyncio.to_thread(self._infer_sync, text)
            label = result["label"].upper()
            raw = float(result["score"])
            normalized = raw if label == "POSITIVE" else (-raw if label == "NEGATIVE" else 0.0)
            return SentimentResult(
                label=label,
                raw_score=raw,
                normalized=normalized,
                text=snippet,
            )
        except Exception as exc:
            log.error("FinBERT inference error: %s", exc)
            return SentimentResult.neutral(snippet)

    def _infer_sync(self, text: str) -> dict:
        outputs = self._pipeline(text)
        # pipeline returns a list with one dict; e.g. [{'label': 'positive', 'score': 0.94}]
        return outputs[0] if isinstance(outputs, list) else outputs

    async def score_batch(self, texts: List[str]) -> List[SentimentResult]:
        """
        Score a batch of texts. More efficient than looping over `score()`.

        Parameters
        ----------
        texts : List of raw text strings.

        Returns
        -------
        List of SentimentResult in the same order as input.
        """
        if not self._loaded:
            return [SentimentResult.neutral(t[:120]) for t in texts]

        try:
            raw_results = await asyncio.to_thread(self._infer_batch_sync, texts)
            results = []
            for i, r in enumerate(raw_results):
                label = r["label"].upper()
                raw = float(r["score"])
                normalized = raw if label == "POSITIVE" else (-raw if label == "NEGATIVE" else 0.0)
                results.append(SentimentResult(
                    label=label,
                    raw_score=raw,
                    normalized=normalized,
                    text=texts[i][:120],
                ))
            return results
        except Exception as exc:
            log.error("FinBERT batch inference error: %s", exc)
            return [SentimentResult.neutral(t[:120]) for t in texts]

    def _infer_batch_sync(self, texts: List[str]) -> List[dict]:
        return self._pipeline(texts, batch_size=16)
