"""Loads model artifacts deterministically and produces anomaly scores."""
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.features import RAW_FEATURES, engineer_features
from src.io_utils import read_json, sha256_file

logger = logging.getLogger("fraud_api.model")


class ModelService:
    """Wraps the Isolation Forest bundle.

    Loading is deterministic: the model file's SHA-256 must match the value
    recorded in metadata.json, otherwise the model is refused.
    """

    def __init__(self, model_dir: Path, threshold_override: float | None = None) -> None:
        self.model_dir = Path(model_dir)
        self.threshold_override = threshold_override
        self.model: Any = None
        self.metadata: dict[str, Any] = {}
        self.metrics: dict[str, Any] = {}
        self.reference_stats: dict[str, Any] | None = None
        self.load_error: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    @property
    def threshold(self) -> float:
        if self.threshold_override is not None:
            return float(self.threshold_override)
        return float(self.metadata["threshold"])

    @property
    def model_version(self) -> str:
        return str(self.metadata.get("model_version", "unknown"))

    def load(self) -> bool:
        """Try to load artifacts. Never raises; sets ``load_error`` on failure."""
        self.model, self.metadata, self.metrics, self.reference_stats = None, {}, {}, None
        try:
            model_path = self.model_dir / "model.joblib"
            meta_path = self.model_dir / "metadata.json"
            if not model_path.exists() or not meta_path.exists():
                self.load_error = "model artifacts not found"
                logger.warning("model_artifacts_missing", extra={"fields": {"model_dir": str(self.model_dir)}})
                return False
            metadata = read_json(meta_path)
            if sha256_file(model_path) != metadata.get("model_sha256"):
                self.load_error = "model checksum mismatch"
                logger.error("model_checksum_mismatch")
                return False
            bundle = joblib.load(model_path)
            self.model = bundle["model"]
            self.metadata = metadata
            metrics_path = self.model_dir / "metrics.json"
            ref_path = self.model_dir / "reference_stats.json"
            self.metrics = read_json(metrics_path) if metrics_path.exists() else {}
            self.reference_stats = read_json(ref_path) if ref_path.exists() else None
            self.load_error = None
            logger.info("model_loaded", extra={"fields": {"model_version": self.model_version}})
            return True
        except Exception:  # noqa: BLE001 - never crash the API on a bad artifact
            self.model = None
            self.load_error = "model failed to load"
            logger.exception("model_load_failed")
            return False

    def predict(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Score transactions. Raises RuntimeError if no model is loaded."""
        if not self.is_loaded:
            raise RuntimeError("model not loaded")
        frame = pd.DataFrame(records)
        features = engineer_features(frame[RAW_FEATURES])
        scores = -self.model.score_samples(features)
        threshold = self.threshold
        results = []
        for rec, score in zip(records, scores):
            suspicious = bool(score >= threshold)
            results.append({
                "transaction_id": rec.get("transaction_id") or f"txn-{uuid.uuid4().hex[:12]}",
                "anomaly_score": round(float(score), 6),
                "is_suspicious": suspicious,
                "classification": "suspicious" if suspicious else "normal",
            })
        return results
