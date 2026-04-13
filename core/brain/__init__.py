"""NEXUS-II — Brain: multi-market factor models, registry, optional FinRL."""

from core.brain.finrl_trainer import FinRLTrainer, FinRLTrainerConfig
from core.brain.model_registry import ModelKind, ModelRegistry, RegisteredModel
from core.brain.multi_market_qlib import (
    MARKET_BSE,
    MARKET_NSE,
    MARKET_US,
    MultiMarketQlibEngine,
)

__all__ = [
    "FinRLTrainer",
    "FinRLTrainerConfig",
    "MARKET_BSE",
    "MARKET_NSE",
    "MARKET_US",
    "ModelKind",
    "ModelRegistry",
    "MultiMarketQlibEngine",
    "RegisteredModel",
]
