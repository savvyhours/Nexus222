"""
NEXUS-II — FinRL training scaffold (Month 3+).

FinRL is not listed in ``requirements.txt`` by default. When you add
reinforcement-learning execution simulation, install FinRL and subclass
or replace ``FinRLTrainer.run`` with your env + algorithm.

See: https://github.com/AI4Finance-Foundation/FinRL
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)


@dataclass
class FinRLTrainerConfig:
    """Minimal configuration for a future FinRL training job."""

    experiment_name: str = "nexus_finrl_v0"
    market: str = "NSE"
    universe_key: str = "NIFTY50"
    total_timesteps: int = 100_000
    seed: int = 42
    checkpoint_dir: Path = field(default_factory=lambda: Path("data/finrl_checkpoints"))
    env_kwargs: dict[str, Any] = field(default_factory=dict)
    algo: str = "PPO"  # e.g. PPO, A2C, SAC


class FinRLTrainer:
    """
    Optional trainer entrypoint. Call ``run()`` after installing finrl and
    pointing ``config.checkpoint_dir`` at a writable location.
    """

    def __init__(self, config: Optional[FinRLTrainerConfig] = None) -> None:
        self.config = config or FinRLTrainerConfig()

    def run(self) -> dict[str, Any]:
        """
        Execute a training run. Raises ``RuntimeError`` if FinRL is unavailable.

        Returns
        -------
        dict
            Metadata about the run (paths, algo, timesteps) for ``ModelRegistry``
            registration of ``ModelKind.FINRL`` entries.
        """
        try:
            import finrl  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "FinRL is not installed. Add it to your environment "
                "(e.g. pip install finrl) and configure data + gym env, "
                "then implement FinRLTrainer.run for NEXUS-II."
            ) from e

        _ = finrl
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log.warning(
            "FinRLTrainer.run: stub only — implement env creation and stable-baselines "
            "training for experiment %r",
            self.config.experiment_name,
        )
        return {
            "status": "stub",
            "experiment_name": self.config.experiment_name,
            "checkpoint_dir": str(self.config.checkpoint_dir.resolve()),
            "algo": self.config.algo,
            "total_timesteps": self.config.total_timesteps,
            "market": self.config.market,
            "universe_key": self.config.universe_key,
        }
