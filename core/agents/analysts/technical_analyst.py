"""
NEXUS-II — Technical Analyst (Tier 1)

Analyses OHLCV data and pre-computed indicators using Claude Sonnet.
Covers:
  - Momentum  : RSI(14), MACD, EMA ribbon (8/13/21/34/55)
  - Trend     : ADX(14), EMA(20/50) crossover, 52-week high breakout
  - Volatility: ATR, Bollinger Bands (20, 2σ), BB squeeze
  - Volume    : OBV, VWAP, volume spike (> 2× 20-bar avg)
  - Intraday  : VWAP Reversal, ORB (Opening Range Breakout 15-min), Gap-and-Go
  - Pivots    : CPR (Central Pivot Range) S/R levels
  - Candlestick composites (backtested on 18.5yr Nifty):
      EMA Ribbon Cross, RSI Divergence + Hammer, Three White Soldiers,
      Engulfing + Volume Spike, Morning Star + RSI < 30

Output: AnalystReport with BUY / SELL / HOLD bias and 0–1 conviction.
"""
from __future__ import annotations

import json
import logging

from core.agents.analysts import AnalystReport, BaseAnalyst
from core.agents.base_agent import Action

log = logging.getLogger(__name__)

# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are the Technical Analyst for NEXUS-II, an AI trading system for Indian markets (NSE/BSE).

Your job: Given a set of technical indicators and OHLCV data for a single stock, synthesise the full technical picture and return a structured JSON assessment.

== INDICATORS TO ANALYSE ==

MOMENTUM:
- RSI(14): < 30 oversold (bullish reversal), > 70 overbought (bearish reversal), divergence is strongest signal
- MACD (12/26/9): Signal line crossover direction and histogram momentum
- EMA ribbon (8/13/21/34/55): All aligned up = strong bull, all down = strong bear, tangled = range-bound

TREND:
- ADX(14): < 20 no trend, 20–25 developing, > 25 confirmed trend, > 40 strong trend
- EMA(20) vs EMA(50): Golden cross / death cross
- 52-week high proximity: < 3% from high = breakout zone

VOLATILITY:
- ATR(14): Absolute volatility measure — compare to 20-day ATR average
- Bollinger Bands(20, 2σ): Price at upper/lower band, BB squeeze (narrow bands = coiling)

VOLUME:
- OBV: Trend confirmation — rising OBV on rising price = healthy, divergence = warning
- VWAP: Price > VWAP bullish intraday, price < VWAP bearish intraday
- Volume spike: Current volume vs 20-bar average (> 2× is significant)

INTRADAY PATTERNS:
- Gap-and-Go: Gap up > 0.5% with volume > 2× → momentum continuation
- ORB: 15-min opening range breakout — price above/below range with volume
- VWAP Reversal: Sustained deviation from VWAP → reversion signal
- CPR: Price above/below Central Pivot Range

CANDLESTICK COMPOSITES (high probability on Indian markets):
- EMA Ribbon Cross (8/13/21/34/55) — PF 6.72, WR 77.8% — strongest trend signal
- RSI Divergence + Hammer — PF 8.51, WR 73.7% — strongest reversal
- Three White Soldiers — PF 2.90, WR 68.9% — bullish continuation
- Engulfing + Volume Spike — PF 2.45, WR 65.2% — reversal
- Morning Star + RSI < 30 — PF 2.12, WR 62.1% — oversold reversal

== SIGNAL RULES ==
- CRISIS/HIGH_VOL regime: Raise conviction bar; only > 0.80 triggers a BUY
- Only one dominant signal direction; if mixed, choose HOLD
- Conviction = weighted average of signal strength (0.0–1.0)
- Candlestick composite signals add 0.10–0.20 to conviction

== OUTPUT FORMAT ==
Return ONLY valid JSON (no markdown):
{
  "signal": "BUY" | "SELL" | "HOLD",
  "conviction": <float 0.0–1.0>,
  "summary": "<2–3 sentences synthesising the technical picture>",
  "key_findings": ["<finding 1>", "<finding 2>", "<finding 3 optional>"],
  "patterns_detected": ["<pattern name>", ...]
}
"""

# ── TechnicalAnalyst ───────────────────────────────────────────────────────────

class TechnicalAnalyst(BaseAnalyst):
    """
    Tier-1 Technical Analyst.

    Expects market_data to contain:
        indicators: dict with keys:
            rsi, macd_line, macd_signal, macd_hist,
            ema_8, ema_13, ema_21, ema_34, ema_55,
            ema_20, ema_50, adx, atr, atr_avg_20,
            bb_upper, bb_mid, bb_lower, bb_width,
            obv, vwap, volume, volume_avg_20,
            high_52w, low_52w,
            cpr_pivot, cpr_bc, cpr_tc     (optional)
            orb_high, orb_low             (optional — 15-min ORB)
        ohlcv: list of dicts [{open, high, low, close, volume}, ...]  (last 5 bars min)
        quote: dict with ltp, open, prev_close
    """

    ANALYST_NAME = "technical"

    async def analyze(self, symbol: str, market_data: dict) -> AnalystReport:
        ind = market_data.get("indicators", {})
        quote = market_data.get("quote", {})
        ohlcv = market_data.get("ohlcv", [])

        raw_data = {
            "rsi":           ind.get("rsi"),
            "macd_line":     ind.get("macd_line"),
            "macd_signal":   ind.get("macd_signal"),
            "macd_hist":     ind.get("macd_hist"),
            "ema_8":         ind.get("ema_8"),
            "ema_13":        ind.get("ema_13"),
            "ema_21":        ind.get("ema_21"),
            "ema_34":        ind.get("ema_34"),
            "ema_55":        ind.get("ema_55"),
            "ema_20":        ind.get("ema_20"),
            "ema_50":        ind.get("ema_50"),
            "adx":           ind.get("adx"),
            "atr":           ind.get("atr"),
            "atr_avg_20":    ind.get("atr_avg_20"),
            "bb_upper":      ind.get("bb_upper"),
            "bb_lower":      ind.get("bb_lower"),
            "bb_width":      ind.get("bb_width"),
            "obv":           ind.get("obv"),
            "vwap":          ind.get("vwap"),
            "volume":        ind.get("volume"),
            "volume_avg_20": ind.get("volume_avg_20"),
            "ltp":           quote.get("ltp"),
            "open":          quote.get("open"),
            "prev_close":    quote.get("prev_close"),
            "high_52w":      ind.get("high_52w"),
            "last_5_bars":   ohlcv[-5:] if len(ohlcv) >= 5 else ohlcv,
            # optional
            "cpr_pivot":     ind.get("cpr_pivot"),
            "orb_high":      ind.get("orb_high"),
            "orb_low":       ind.get("orb_low"),
        }

        user_content = (
            f"Analyse the technical picture for **{symbol}**.\n\n"
            f"```json\n{json.dumps(raw_data, indent=2, default=str)}\n```"
        )

        try:
            raw_text = await self._call_claude(_SYSTEM_PROMPT, user_content)
            parsed = json.loads(self._strip_fences(raw_text))

            signal = Action(parsed.get("signal", "HOLD").upper())
            conviction = float(parsed.get("conviction", 0.0))
            summary = parsed.get("summary", "")
            key_findings = parsed.get("key_findings", [])
            patterns = parsed.get("patterns_detected", [])

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
            metadata={"patterns_detected": patterns},
        )
