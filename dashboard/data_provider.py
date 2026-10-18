"""Unified resilient data layer for the Fraud Sentinel Dashboard.

Supports both:
1. Production REST API Mode (HTTP to FastAPI at API_BASE_URL)
2. Standalone Autonomous Engine Mode (embedded ModelService + Database fallback
   so the dashboard never crashes even if the backend server is stopped or another app runs on port 8000).
"""
from __future__ import annotations

import logging
import os
import sys
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

# Ensure repository root is in sys.path for direct imports if needed
DASHBOARD_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DASHBOARD_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("sentinel.provider")

DEFAULT_API_URL = os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = 4.0


class DataProvider:
    def __init__(self, api_url: str = DEFAULT_API_URL) -> None:
        self.api_url = api_url.rstrip("/")
        self.is_live_api: bool = False
        self.local_model_service: Any = None
        self.local_database: Any = None
        self.local_monitoring: Any = None
        self._init_local_fallback()
        self.check_api_connectivity()

    def set_api_url(self, new_url: str) -> bool:
        """Updates API target URL and re-probes connectivity."""
        self.api_url = new_url.rstrip("/")
        return self.check_api_connectivity()

    def _init_local_fallback(self) -> None:
        """Initializes direct local model & db instances as zero-crash fallback."""
        try:
            from app.config import Settings
            from app.database import Database
            from app.model_service import ModelService
            from app.monitoring import MonitoringService

            model_dir = PROJECT_ROOT / "models"
            db_path = PROJECT_ROOT / "data" / "fraud_logs.db"

            self.local_database = Database(db_path)
            self.local_database.init()

            self.local_model_service = ModelService(model_dir)
            self.local_model_service.load()

            settings = replace(Settings.from_env(), model_dir=model_dir, db_path=db_path)
            self.local_monitoring = MonitoringService(self.local_database, self.local_model_service, settings)
        except Exception as exc:
            logger.warning(f"Local fallback initialization: {exc}")

    def check_api_connectivity(self) -> bool:
        """Pings /health and verifies this is the genuine Fraud Detection API."""
        try:
            resp = requests.get(f"{self.api_url}/health", timeout=1.8)
            if resp.status_code == 200:
                data = resp.json()
                # Verify it is our fraud detection API, not another service on port 8000
                if "model_loaded" in data or "model_version" in data:
                    self.is_live_api = True
                    return True
        except requests.RequestException:
            pass
        self.is_live_api = False
        return False

    # ------------------------------------------------------------- GETTERS
    def get_health(self) -> dict[str, Any]:
        if self.is_live_api:
            try:
                resp = requests.get(f"{self.api_url}/health", timeout=TIMEOUT)
                if resp.status_code == 200:
                    return resp.json()
            except requests.RequestException:
                self.is_live_api = False

        if self.local_model_service and self.local_model_service.is_loaded:
            return {
                "status": "healthy",
                "model_loaded": True,
                "model_version": self.local_model_service.model_version,
                "engine_mode": "embedded_fallback",
            }
        return {"status": "unhealthy", "model_loaded": False, "detail": "API offline and local model missing."}

    def get_metrics(self) -> dict[str, Any]:
        if self.is_live_api:
            try:
                resp = requests.get(f"{self.api_url}/metrics", timeout=TIMEOUT)
                if resp.status_code == 200:
                    return resp.json()
            except requests.RequestException:
                self.is_live_api = False

        if self.local_monitoring:
            try:
                return self.local_monitoring.metrics()
            except Exception:
                pass
        return {"requests": {"total": 0, "errors": 0}, "predictions": {"total": 0, "suspicious": 0, "suspicious_rate": 0}}

    def get_model_info(self) -> dict[str, Any]:
        if self.is_live_api:
            try:
                resp = requests.get(f"{self.api_url}/model-info", timeout=TIMEOUT)
                if resp.status_code == 200:
                    return resp.json()
            except requests.RequestException:
                self.is_live_api = False

        if self.local_model_service:
            md = self.local_model_service.metadata
            from src.features import FEATURE_DESCRIPTIONS
            return {
                "model_type": md.get("model_type", "IsolationForest"),
                "model_version": self.local_model_service.model_version,
                "threshold": self.local_model_service.threshold,
                "score_definition": md.get("score_definition", ""),
                "feature_schema": md.get("raw_feature_schema", FEATURE_DESCRIPTIONS),
                "model_features": md.get("model_features", []),
                "training_metadata": {
                    k: md.get(k) for k in (
                        "trained_at_utc", "data_source", "is_synthetic_data", "random_seed",
                        "hyperparameters", "training_rows", "threshold_selection", "sklearn_version",
                    )
                },
                "evaluation": self.local_model_service.metrics,
            }
        return {}

    def get_drift(self) -> dict[str, Any]:
        if self.is_live_api:
            try:
                resp = requests.get(f"{self.api_url}/drift", timeout=TIMEOUT)
                if resp.status_code == 200:
                    return resp.json()
            except requests.RequestException:
                self.is_live_api = False

        if self.local_monitoring:
            try:
                return self.local_monitoring.drift()
            except Exception:
                pass
        return {"status": "unavailable", "message": "Drift telemetry offline", "features": {}}

    def get_recent_predictions(self, limit: int = 300) -> list[dict[str, Any]]:
        if self.is_live_api:
            try:
                resp = requests.get(f"{self.api_url}/predictions/recent", params={"limit": limit}, timeout=TIMEOUT)
                if resp.status_code == 200:
                    return resp.json().get("predictions", [])
            except requests.RequestException:
                self.is_live_api = False

        if self.local_database:
            try:
                return self.local_database.recent_predictions(limit)
            except Exception:
                pass
        return []

    # ------------------------------------------------------------- ACTIONS
    def predict(self, payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
        """Scores a transaction via API or local model."""
        if self.is_live_api:
            try:
                resp = requests.post(f"{self.api_url}/predict", json=payload, timeout=TIMEOUT)
                if resp.status_code == 422:
                    return None, f"Validation error: {resp.json().get('errors')}"
                resp.raise_for_status()
                return resp.json(), None
            except requests.RequestException as exc:
                logger.warning(f"API prediction call failed: {exc}, switching to local fallback")
                self.is_live_api = False

        if self.local_model_service and self.local_model_service.is_loaded:
            start = time.perf_counter()
            results = self.local_model_service.predict([payload])
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            res = results[0]
            res.update({
                "threshold": self.local_model_service.threshold,
                "model_version": self.local_model_service.model_version,
                "processing_time_ms": round(elapsed_ms, 3),
                "request_id": f"local-{uuid.uuid4().hex[:12]}",
            })
            if self.local_database:
                try:
                    self.local_database.log_predictions(
                        res["request_id"], [payload], results,
                        self.local_model_service.threshold, self.local_model_service.model_version, elapsed_ms
                    )
                except Exception:
                    pass
            return res, None

        return None, "Both API and local model service are unavailable."

    def generate_demo_traffic(self, n: int = 50, drift: bool = False) -> tuple[dict[str, Any] | None, str | None]:
        """Generates synthetic demo transactions."""
        if self.is_live_api:
            try:
                resp = requests.post(f"{self.api_url}/demo/generate", json={"n": n, "drift": drift}, timeout=TIMEOUT)
                resp.raise_for_status()
                return resp.json(), None
            except requests.RequestException:
                self.is_live_api = False

        try:
            from src.data_generator import generate_demo_transactions
            records = generate_demo_transactions(n, drift=drift)
            if self.local_model_service and self.local_model_service.is_loaded:
                start = time.perf_counter()
                results = self.local_model_service.predict(records)
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                req_id = f"demo-{uuid.uuid4().hex[:12]}"
                if self.local_database:
                    self.local_database.log_predictions(
                        req_id, records, results,
                        self.local_model_service.threshold, self.local_model_service.model_version, elapsed_ms
                    )
                suspicious_count = sum(1 for r in results if r["is_suspicious"])
                return {
                    "generated": len(results),
                    "suspicious": suspicious_count,
                    "drift_simulated": drift,
                    "synthetic": True,
                    "request_id": req_id,
                }, None
        except Exception as exc:
            return None, str(exc)

        return None, "Unable to generate demo traffic."


def calculate_feature_attributions(
    txn: dict[str, Any],
    ref_stats: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Calculates explainable risk factor attributions for a given transaction."""
    amt = float(txn.get("amount", 30.0))
    avg = float(txn.get("avg_amount_30d", 30.0))
    dist = float(txn.get("distance_from_home_km", 5.0))
    count24 = float(txn.get("txn_count_24h", 2.0))
    hour = int(txn.get("hour", 12))
    is_foreign = int(txn.get("is_foreign", 0))
    is_online = int(txn.get("is_online", 0))
    card_age = float(txn.get("card_age_days", 400.0))
    merchant = int(txn.get("merchant_category", 2))

    ratio = amt / max(avg, 1.0)
    attributions = []

    # 1. Distance anomaly
    dist_impact = min(100.0, (dist / 250.0) * 45.0) if dist > 25.0 else -15.0
    attributions.append({"feature_name": "Geographic Distance", "impact": round(dist_impact, 1)})

    # 2. Amount to 30d Avg Ratio
    ratio_impact = min(100.0, (ratio - 1.0) * 25.0) if ratio > 1.8 else -10.0
    attributions.append({"feature_name": "Spend vs 30d Avg Ratio", "impact": round(ratio_impact, 1)})

    # 3. 24h Velocity Spike
    vel_impact = min(100.0, (count24 - 3.0) * 18.0) if count24 > 4 else -12.0
    attributions.append({"feature_name": "24h Velocity Spike", "impact": round(vel_impact, 1)})

    # 4. Circadian / Night factor (0-5 AM)
    night_impact = 35.0 if (hour <= 5 or hour >= 23) else -15.0
    attributions.append({"feature_name": "Circadian Night Risk", "impact": round(night_impact, 1)})

    # 5. Cross-border Foreign flag
    foreign_impact = 45.0 if is_foreign else -10.0
    attributions.append({"feature_name": "Cross-Border / Foreign", "impact": round(foreign_impact, 1)})

    # 6. High-risk merchant (7, 8, 9)
    merchant_impact = 30.0 if merchant in (7, 8, 9) else -8.0
    attributions.append({"feature_name": "High-Risk Merchant Code", "impact": round(merchant_impact, 1)})

    # 7. Card age risk (new cards < 30 days have higher risk)
    age_impact = 25.0 if card_age < 30 else -10.0
    attributions.append({"feature_name": "New Card Profile", "impact": round(age_impact, 1)})

    return attributions
