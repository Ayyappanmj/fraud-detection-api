"""Application settings, read from environment variables (no secrets needed)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    model_dir: Path
    db_path: Path
    threshold_override: float | None
    drift_psi_threshold: float
    drift_min_samples: int
    drift_window: int
    max_batch_size: int = 1000
    app_version: str = "1.0.0"

    @classmethod
    def from_env(cls) -> "Settings":
        override = os.getenv("DECISION_THRESHOLD")
        try:
            threshold_override = float(override) if override else None
        except ValueError:
            threshold_override = None
        return cls(
            model_dir=Path(os.getenv("MODEL_DIR", "models")),
            db_path=Path(os.getenv("DB_PATH", "data/fraud_logs.db")),
            threshold_override=threshold_override,
            drift_psi_threshold=_float_env("DRIFT_PSI_THRESHOLD", 0.2),
            drift_min_samples=_int_env("DRIFT_MIN_SAMPLES", 100),
            drift_window=_int_env("DRIFT_WINDOW", 500),
        )
