"""Structured JSON logging, metrics summaries and drift status."""
from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from app.config import Settings
from app.database import Database
from app.model_service import ModelService
from src.drift_check import DRIFT_DISCLAIMER, compute_drift
from src.features import RAW_FEATURES

MIN_LATENCY_OBSERVATIONS = 20


class JsonFormatter(logging.Formatter):
    """One JSON object per log line (easy to grep and to ship to log tools)."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(getattr(record, "fields", {}))
        if record.exc_info:  # stack traces go to server logs only, never to clients
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(level: str = "INFO") -> None:
    root = logging.getLogger("fraud_api")
    root.setLevel(level.upper())
    if not any(getattr(h, "_fraud_api", False) for h in root.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        handler._fraud_api = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    root.propagate = False


def latency_summary(values: list[float], min_obs: int = MIN_LATENCY_OBSERVATIONS) -> dict[str, Any]:
    """Mean/percentiles; percentiles only reported with enough observations."""
    summary: dict[str, Any] = {"count": len(values), "mean_ms": None, "p50_ms": None, "p95_ms": None, "p99_ms": None}
    if values:
        summary["mean_ms"] = round(float(np.mean(values)), 3)
    if len(values) >= min_obs:
        p50, p95, p99 = np.percentile(values, [50, 95, 99])
        summary.update(p50_ms=round(float(p50), 3), p95_ms=round(float(p95), 3), p99_ms=round(float(p99), 3))
    else:
        summary["note"] = f"Percentiles need at least {min_obs} observations."
    return summary


class MonitoringService:
    def __init__(self, db: Database, model_service: ModelService, settings: Settings) -> None:
        self.db = db
        self.model_service = model_service
        self.settings = settings
        self.started_at = time.time()

    def metrics(self) -> dict[str, Any]:
        req = self.db.request_stats()
        preds = self.db.prediction_counts()
        total_req = req["total"]
        return {
            "requests": {
                "total": total_req,
                "errors": req["errors"],
                "error_rate": round(req["errors"] / total_req, 4) if total_req else 0.0,
                "by_path": req["by_path"],
            },
            "predictions": {
                "total": preds["total"],
                "suspicious": preds["suspicious"],
                "suspicious_rate": round(preds["suspicious"] / preds["total"], 4) if preds["total"] else 0.0,
                "synthetic": preds["synthetic"],
            },
            "latency_ms": latency_summary(self.db.prediction_latencies()),
            "uptime_seconds": round(time.time() - self.started_at, 1),
            "storage_note": "Counts come from a local SQLite file; on Render free tier it resets on restart/redeploy.",
        }

    def drift(self) -> dict[str, Any]:
        reference = self.model_service.reference_stats
        if reference is None:
            return {
                "status": "unavailable",
                "checked_at": datetime.now(timezone.utc).isoformat(),
                "message": "Reference statistics are not available (model not loaded).",
                "features": {},
                "drifted_features": [],
                "disclaimer": DRIFT_DISCLAIMER,
            }
        rows = self.db.recent_predictions(limit=self.settings.drift_window)
        recent = pd.DataFrame(rows)[RAW_FEATURES] if rows else pd.DataFrame(columns=RAW_FEATURES)
        return compute_drift(
            reference, recent,
            psi_threshold=self.settings.drift_psi_threshold,
            min_samples=self.settings.drift_min_samples,
        )
