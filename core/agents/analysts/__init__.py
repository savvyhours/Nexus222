"""
NEXUS-II — Tier-1 Analyst Agents
Defines the AnalystReport dataclass and BaseAnalyst ABC shared by all four
analyst implementations: Technical, Sentiment, Fundamental, Macro.

These analysts produce structured reports consumed by the Tier-2 researchers
(Bull / Bear / Risk) and then the Debate Arena before reaching the Portfolio Manager.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Optional
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    import anthropic
    from core.calibration.weight_calibration_agent import WeightCalibrationAgent

from core.agents.base_agent import Action

log = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")


# ── Shared data model ──────────────────────────────────────────────────────────

@dataclass
class AnalystReport:
    """
    Canonical report emitted by every Tier-1 analyst.

    Consumed by Bull / Bear / Risk researchers (Tier 2) and formatted into the
    Debate Arena prompt (Tier 3).

    Fields
    ------
    analyst       : which analyst produced this ("technical" | "sentiment" |
                    "fundamental" | "macro")
    symbol        : NSE/BSE ticker (e.g. "RELIANCE")
    signal        : BUY / SELL / HOLD directional bias
    conviction    : 0.0–1.0 — how strongly the analyst believes the signal
    summary       : 2–3 sentence human-readable synthesis suitable for researchers
    key_findings  : bullet-point list of supporting evidence
    raw_data      : verbatim indicator values fed into the LLM for auditability
    timestamp     : IST wall-clock time of report generation
    metadata      : optional extra fields (e.g. patterns detected, entity list)
    """

    analyst:      str
    symbol:       str
    signal:       Action
    conviction:   float
    summary:      str
    key_findings: list[str]
    raw_data:     dict
    timestamp:    datetime = field(default_factory=lambda: datetime.now(IST))
    metadata:     dict     = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.conviction = max(0.0, min(1.0, self.conviction))

    def to_markdown(self) -> str:
        """Render report as Markdown for use in researcher / debate prompts."""
        bullets = "\n".join(f"  - {f}" for f in self.key_findings)
        return (
            f"### {self.analyst.title()} Analyst — {self.symbol}\n"
            f"**Signal:** {self.signal.value}  |  **Conviction:** {self.conviction:.0%}\n\n"
            f"{self.summary}\n\n"
            f"**Key Findings:**\n{bullets}\n"
        )


# ── BaseAnalyst ────────────────────────────────────────────────────────────────

class BaseAnalyst(ABC):
    """
    Abstract base for all four Tier-1 analyst agents.

    Subclasses must implement `analyze()` to produce an AnalystReport.
    All primary LLM calls target Claude Sonnet (LLM_MAIN).
    High-throughput NLP subtasks (entity extraction) use Claude Haiku (LLM_FAST).
    """

    #: Must match the analyst key used in AnalystReport.analyst
    ANALYST_NAME: str = ""

    def __init__(
        self,
        claude_client: "anthropic.AsyncAnthropic",
        calibration_agent: "WeightCalibrationAgent",
    ) -> None:
        self._claude = claude_client
        self._calibration = calibration_agent
        self.name: str = self.ANALYST_NAME or self.__class__.__name__
        log.info("Analyst '%s' initialised", self.name)

    # ── Abstract interface ─────────────────────────────────────────────────────

    @abstractmethod
    async def analyze(self, symbol: str, market_data: dict) -> AnalystReport:
        """
        Produce an analyst report for `symbol`.

        Parameters
        ----------
        symbol      : NSE/BSE ticker (e.g. "RELIANCE")
        market_data : dict containing at minimum the keys relevant to this analyst
                      type (ohlcv, indicators, quote, depth, sentiment,
                      fundamentals, macro).  Extra keys are ignored.

        Returns
        -------
        AnalystReport — always returned; conviction=0.0 / signal=HOLD on error.
        """

    # ── Shared LLM helpers ─────────────────────────────────────────────────────

    async def _call_claude(
        self,
        system_prompt: str,
        user_content: str,
        max_tokens: int = 1024,
    ) -> str:
        """Call Claude Sonnet (LLM_MAIN) and return the raw text."""
        from config.settings import LLM_MAIN

        response = await self._claude.messages.create(
            model=LLM_MAIN,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}],
        )
        return response.content[0].text.strip()

    async def _call_claude_fast(
        self,
        system_prompt: str,
        user_content: str,
        max_tokens: int = 512,
    ) -> str:
        """Call Claude Haiku (LLM_FAST) for cheap, high-throughput NLP tasks."""
        from config.settings import LLM_FAST

        response = await self._claude.messages.create(
            model=LLM_FAST,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}],
        )
        return response.content[0].text.strip()

    @staticmethod
    def _strip_fences(text: str) -> str:
        """Remove markdown code fences before JSON parsing."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # drop first line (``` or ```json) and last line (```)
            text = "\n".join(lines[1:-1]).strip()
        return text

    def _fallback_report(self, symbol: str, raw_data: dict, error: str) -> AnalystReport:
        """Return a HOLD/zero-conviction report when analysis fails."""
        log.error("Analyst '%s' failed for %s: %s", self.name, symbol, error)
        return AnalystReport(
            analyst=self.ANALYST_NAME,
            symbol=symbol,
            signal=Action.HOLD,
            conviction=0.0,
            summary=f"Analysis unavailable: {error}",
            key_findings=["LLM call failed — defaulting to HOLD with zero conviction"],
            raw_data=raw_data,
            metadata={"error": error},
        )


# ── Concrete analyst imports (bottom of module — avoids circular import) ───────

from core.agents.analysts.technical_analyst import TechnicalAnalyst      # noqa: E402
from core.agents.analysts.sentiment_analyst import SentimentAnalyst      # noqa: E402
from core.agents.analysts.fundamental_analyst import FundamentalAnalyst  # noqa: E402
from core.agents.analysts.macro_analyst import MacroAnalyst              # noqa: E402

__all__ = [
    "AnalystReport",
    "BaseAnalyst",
    "TechnicalAnalyst",
    "SentimentAnalyst",
    "FundamentalAnalyst",
    "MacroAnalyst",
]
