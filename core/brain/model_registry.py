"""
NEXUS-II — Model registry for Qlib / FinRL artifacts per market.

Persists a small JSON catalog so runtime code can resolve which trained
model (or precomputed prediction file) is active for NSE vs other venues.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Optional

log = logging.getLogger(__name__)

_DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parents[2] / "data" / "model_registry.json"


class ModelKind(str, Enum):
    """Training / inference backend."""

    QLIB_ALPHA158_LGBM = "qlib_alpha158_lgbm"
    QLIB_CUSTOM = "qlib_custom"
    FINRL = "finrl"
    HEURISTIC = "heuristic"


@dataclass
class RegisteredModel:
    """One versioned model or score bundle."""

    model_id: str
    market: str  # e.g. "NSE", "US"
    universe_key: str  # keys from config.universe.MARKET_UNIVERSES
    kind: ModelKind
    artifact_path: str = ""  # qlib recorder dir, pickle, or FinRL checkpoint folder
    predictions_path: str = ""  # optional CSV/Parquet of per-symbol scores (see multi_market_qlib)
    feature_set: str = "alpha158"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    notes: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value if isinstance(self.kind, ModelKind) else str(self.kind)
        return d

    @staticmethod
    def from_json(d: dict[str, Any]) -> "RegisteredModel":
        raw_kind = d.get("kind", ModelKind.QLIB_ALPHA158_LGBM.value)
        try:
            kind = ModelKind(raw_kind)
        except ValueError:
            kind = ModelKind.QLIB_CUSTOM
        return RegisteredModel(
            model_id=d["model_id"],
            market=d["market"],
            universe_key=d.get("universe_key", "NIFTY50"),
            kind=kind,
            artifact_path=d.get("artifact_path", ""),
            predictions_path=d.get("predictions_path", ""),
            feature_set=d.get("feature_set", "alpha158"),
            created_at=d.get("created_at", ""),
            notes=d.get("notes", ""),
            meta=dict(d.get("meta") or {}),
        )


def _registry_path() -> Path:
    raw = os.getenv("NEXUS_MODEL_REGISTRY_PATH", "")
    return Path(raw) if raw else _DEFAULT_REGISTRY_PATH


class ModelRegistry:
    """
    Load / save model catalog and track active model_id per market.

    JSON shape::
        {
          "models": [ { ...RegisteredModel... }, ... ],
          "active_by_market": { "NSE": "nse_alpha158_v1", "US": "..." }
        }
    """

    def __init__(self, path: Optional[Path | str] = None) -> None:
        self.path = Path(path) if path else _registry_path()
        self._models: dict[str, RegisteredModel] = {}
        self._active_by_market: dict[str, str] = {}
        self.load()

    def load(self) -> None:
        self._models.clear()
        self._active_by_market.clear()
        if not self.path.is_file():
            log.info("Model registry not found at %s — starting empty", self.path)
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            log.warning("Could not read model registry %s: %s", self.path, e)
            return
        for m in raw.get("models") or []:
            try:
                rm = RegisteredModel.from_json(m)
                self._models[rm.model_id] = rm
            except KeyError as e:
                log.warning("Skipping invalid registry entry: %s (%s)", m, e)
        self._active_by_market = dict(raw.get("active_by_market") or {})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "models": [m.to_json() for m in self._models.values()],
            "active_by_market": dict(self._active_by_market),
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def register(self, model: RegisteredModel, *, set_active: bool = False) -> None:
        self._models[model.model_id] = model
        if set_active:
            self._active_by_market[model.market] = model.model_id
        self.save()

    def unregister(self, model_id: str) -> None:
        self._models.pop(model_id, None)
        for mkt, mid in list(self._active_by_market.items()):
            if mid == model_id:
                del self._active_by_market[mkt]
        self.save()

    def get(self, model_id: str) -> Optional[RegisteredModel]:
        return self._models.get(model_id)

    def iter_models(self) -> Iterator[RegisteredModel]:
        return iter(self._models.values())

    def models_for_market(self, market: str) -> list[RegisteredModel]:
        return [m for m in self._models.values() if m.market.upper() == market.upper()]

    def set_active(self, market: str, model_id: str) -> None:
        if model_id not in self._models:
            raise KeyError(f"Unknown model_id: {model_id}")
        self._active_by_market[market.upper()] = model_id
        self.save()

    def get_active(self, market: str) -> Optional[RegisteredModel]:
        mid = self._active_by_market.get(market.upper())
        if not mid:
            # fallback: first model for this market
            cands = self.models_for_market(market)
            return cands[0] if len(cands) == 1 else None
        return self._models.get(mid)

    def ensure_default_nse_stub(self) -> None:
        """Create a minimal on-disk registry so imports work out of the box."""
        if self._models:
            return
        stub = RegisteredModel(
            model_id="nse_heuristic_v0",
            market="NSE",
            universe_key="NIFTY50",
            kind=ModelKind.HEURISTIC,
            notes="Placeholder until Qlib artifacts and predictions_path are configured.",
        )
        self._models[stub.model_id] = stub
        self._active_by_market["NSE"] = stub.model_id
        self.save()
