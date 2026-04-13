"""
NEXUS-II — Multi-market Qlib-style factor outputs for QuantAgent.

Resolves the active model from ``ModelRegistry``, optionally loads
precomputed per-symbol scores (CSV), and otherwise derives proxy factors
from ``compute_all_indicators`` fields so ``QuantAgent`` always receives:

  - qlib_rank_percentile
  - qlib_predicted_return_5d
  - momentum_20d_rank
  - quality_score
  - low_vol_rank

Native Qlib online inference is environment-specific; extend
``MultiMarketQlibEngine._predict_qlib_native`` when your provider_uri
and recorder are wired.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from core.brain.model_registry import ModelKind, ModelRegistry, RegisteredModel

log = logging.getLogger(__name__)

MARKET_NSE = "NSE"
MARKET_BSE = "BSE"
MARKET_US = "US"


def _norm_market(market: str) -> str:
    m = (market or MARKET_NSE).strip().upper()
    return m if m else MARKET_NSE


@lru_cache(maxsize=4)
def _load_predictions_table(path_str: str) -> Optional[pd.DataFrame]:
    path = Path(path_str)
    if not path.is_file():
        return None
    try:
        if path.suffix.lower() in (".parquet", ".pq"):
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
    except Exception as e:
        log.warning("Failed to read predictions file %s: %s", path, e)
        return None
    if "symbol" not in df.columns:
        log.warning("predictions file %s missing 'symbol' column", path)
        return None
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    return df.set_index("symbol", drop=False)


def _is_number(v: Any) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


class MultiMarketQlibEngine:
    """
    Produce fundamentals fields consumed by ``QuantAgent`` for a given symbol.

    Parameters
    ----------
    registry
        Shared ``ModelRegistry`` instance; if None, a default path is used.
    prefer_native_qlib
        When True, attempt native Qlib scoring before CSV / heuristics.
    """

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        *,
        prefer_native_qlib: bool = False,
    ) -> None:
        self.registry = registry or ModelRegistry()
        self.prefer_native_qlib = prefer_native_qlib or (
            os.getenv("NEXUS_QLIB_NATIVE", "").lower() in ("1", "true", "yes")
        )
        self.registry.ensure_default_nse_stub()

    def active_model(self, market: str) -> Optional[RegisteredModel]:
        return self.registry.get_active(_norm_market(market))

    def enrich_fundamentals(
        self,
        symbol: str,
        market: str,
        market_data: dict[str, Any],
    ) -> dict[str, float]:
        sym = (symbol or "UNKNOWN").upper().strip()
        mkt = _norm_market(market)
        fund = dict(market_data.get("fundamentals") or {})
        ind = market_data.get("indicators") or {}
        quote = market_data.get("quote") or {}

        model = self.active_model(mkt)
        out: dict[str, float] = {}

        native: Optional[dict[str, float]] = None
        if self.prefer_native_qlib and model and model.kind != ModelKind.HEURISTIC:
            native = self._predict_qlib_native(sym, mkt, model, market_data)

        if native:
            out.update(native)
        else:
            pre = self._from_precomputed(sym, model)
            if pre:
                out.update(pre)
            else:
                out.update(self._indicator_proxy_factors(sym, ind, quote))

        merged = {**fund, **out}
        return {
            k: float(v)
            for k, v in merged.items()
            if isinstance(v, (int, float)) or _is_number(v)
        }

    def _from_precomputed(self, symbol: str, model: Optional[RegisteredModel]) -> dict[str, float]:
        if not model or not model.predictions_path:
            return {}
        df = _load_predictions_table(str(Path(model.predictions_path).resolve()))
        if df is None or symbol not in df.index:
            return {}
        row = df.loc[symbol]
        keys = (
            "qlib_rank_percentile",
            "qlib_predicted_return_5d",
            "momentum_20d_rank",
            "quality_score",
            "low_vol_rank",
        )
        out: dict[str, float] = {}
        for k in keys:
            if k in row.index and pd.notna(row[k]):
                out[k] = float(row[k])
        return out

    @staticmethod
    def _indicator_proxy_factors(
        symbol: str,
        ind: dict[str, Any],
        quote: dict[str, Any],
    ) -> dict[str, float]:
        del symbol  # reserved for future symbol-specific calibrations
        rsi = float(ind.get("rsi", 50.0) or 50.0)
        rsi = max(0.0, min(100.0, rsi))

        ema_bull = 0.0
        for k in ("ema_trend", "ema_stack_bullish"):
            if k in ind:
                try:
                    ema_bull = max(ema_bull, float(ind[k]))
                except (TypeError, ValueError):
                    pass
        ema_bull = max(0.0, min(1.0, ema_bull))
        qlib_rank_pct = float(np.clip(40.0 + (rsi - 50.0) * 0.8 + ema_bull * 25.0, 5.0, 95.0))

        momentum_rank = rsi

        adx = float(ind.get("adx", 20.0) or 20.0)
        quality_score = float(np.clip(0.35 + min(adx, 45.0) / 90.0 + ema_bull * 0.25, 0.1, 0.95))

        atr = float(ind.get("atr", 0.0) or 0.0)
        ltp = float(quote.get("ltp", ind.get("ltp", 0.0)) or 0.0)
        atr_pct = (atr / ltp * 100.0) if ltp > 0 else 3.0
        atr_pct = float(np.clip(atr_pct, 0.05, 15.0))
        low_vol_rank = float(np.clip(100.0 - atr_pct * 6.0, 5.0, 95.0))

        pred_ret = float((qlib_rank_pct - 50.0) / 500.0)

        return {
            "qlib_rank_percentile": qlib_rank_pct,
            "qlib_predicted_return_5d": pred_ret,
            "momentum_20d_rank": momentum_rank,
            "quality_score": quality_score,
            "low_vol_rank": low_vol_rank,
        }

    def _predict_qlib_native(
        self,
        symbol: str,
        market: str,
        model: RegisteredModel,
        market_data: dict[str, Any],
    ) -> Optional[dict[str, float]]:
        """
        Hook for real Qlib scoring. Implement using your ``provider_uri``,
        dataset config, and saved experiment / model bundle.

        Returns None to fall back to precomputed CSV or heuristics.
        """
        try:
            import qlib  # noqa: F401
        except ImportError:
            return None
        _ = (symbol, market, model, market_data)
        return None
