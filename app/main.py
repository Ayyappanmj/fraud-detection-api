"""FastAPI application factory.

Run locally:
    uvicorn app.main:create_app --factory --reload --port 8000

DEMO ONLY: this is not a production payment-authorisation system.
"""
from __future__ import annotations

import logging
import os
import re
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.config import Settings
from app.database import Database
from app.model_service import ModelService
from app.monitoring import MonitoringService, setup_logging
from app.schemas import (
    BatchRequest,
    BatchResponse,
    DemoRequest,
    DemoResponse,
    PredictionItem,
    PredictionResponse,
    Transaction,
)
from src.data_generator import generate_demo_transactions
from src.features import FEATURE_DESCRIPTIONS

logger = logging.getLogger("fraud_api")

PROJECT_NAME = "Real-Time Fraud Detection API"
DISCLAIMER = (
    "Educational portfolio demo trained on SYNTHETIC data. Anomaly scores are NOT fraud probabilities and "
    "this service must not be used to authorise, decline or investigate real payments."
)
# Polled frequently by Render / the dashboard; keep them out of the request log.
_UNLOGGED_PATHS = {
    "/health", "/metrics", "/drift", "/model-info", "/predictions/recent",
    "/docs", "/redoc", "/openapi.json", "/favicon.ico",
}
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


def _request_id(header_value: str | None) -> str:
    if header_value and _REQUEST_ID_RE.match(header_value):
        return header_value
    return uuid.uuid4().hex


def create_app(model_dir: Path | str | None = None, db_path: Path | str | None = None) -> FastAPI:
    settings = Settings.from_env()
    if model_dir is not None:
        settings = replace(settings, model_dir=Path(model_dir))
    if db_path is not None:
        settings = replace(settings, db_path=Path(db_path))

    setup_logging(os.getenv("LOG_LEVEL", "INFO"))

    app = FastAPI(
        title=PROJECT_NAME,
        version=settings.app_version,
        description=f"{DISCLAIMER}\n\nInteractive docs: /docs",
    )

    db = Database(settings.db_path)
    db.init()
    model_service = ModelService(settings.model_dir, settings.threshold_override)
    model_service.load()
    app.state.settings = settings
    app.state.db = db
    app.state.model_service = model_service
    app.state.monitoring = MonitoringService(db, model_service, settings)

    # ------------------------------------------------------------------ middleware
    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = _request_id(request.headers.get("X-Request-ID"))
        request.state.request_id = rid
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:  # noqa: BLE001 - convert ANY failure into a safe 500
            logger.exception("unhandled_error", extra={"fields": {"request_id": rid, "path": request.url.path}})
            response = JSONResponse(status_code=500, content={"detail": "Internal server error", "request_id": rid})
        latency_ms = (time.perf_counter() - start) * 1000.0
        response.headers["X-Request-ID"] = rid
        path = request.url.path
        if path not in _UNLOGGED_PATHS:
            try:
                db.log_request(rid, request.method, path, response.status_code, latency_ms)
            except Exception:  # noqa: BLE001 - logging must never break the API
                logger.warning("request_log_failed", extra={"fields": {"request_id": rid}})
        logger.info("request", extra={"fields": {
            "request_id": rid, "method": request.method, "path": path,
            "status": response.status_code, "latency_ms": round(latency_ms, 2),
        }})
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Only field names and messages are returned; submitted values are NOT echoed back.
        errors = [
            {
                "field": ".".join(str(p) for p in err.get("loc", ()) if p != "body"),
                "message": err.get("msg", "invalid value"),
                "type": err.get("type", "value_error"),
            }
            for err in exc.errors()[:20]
        ]
        rid = getattr(request.state, "request_id", "unknown")
        return JSONResponse(status_code=422, content={"detail": "Request validation failed", "errors": errors, "request_id": rid})

    # ------------------------------------------------------------------ helpers
    def require_model(request: Request) -> ModelService:
        service: ModelService = request.app.state.model_service
        if not service.is_loaded:
            raise HTTPException(status_code=503, detail="Model is not available. Train it first: python -m src.train")
        return service

    def score_and_log(request: Request, service: ModelService, records: list[dict[str, Any]]):
        start = time.perf_counter()
        results = service.predict(records)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        try:
            request.app.state.db.log_predictions(
                request.state.request_id, records, results, service.threshold, service.model_version, elapsed_ms
            )
        except Exception:  # noqa: BLE001
            logger.warning("prediction_log_failed", extra={"fields": {"request_id": request.state.request_id}})
        return results, elapsed_ms

    # ------------------------------------------------------------------ endpoints
    @app.get("/", tags=["info"])
    def root() -> dict[str, Any]:
        return {
            "message": f"Welcome to the {PROJECT_NAME}",
            "version": settings.app_version,
            "docs": "/docs",
            "endpoints": ["/health", "/model-info", "/predict", "/predict/batch", "/metrics", "/drift"],
            "disclaimer": DISCLAIMER,
        }

    @app.get("/health", tags=["info"])
    def health(request: Request) -> JSONResponse:
        service: ModelService = request.app.state.model_service
        body = {
            "status": "healthy" if service.is_loaded else "unhealthy",
            "model_loaded": service.is_loaded,
            "model_version": service.model_version if service.is_loaded else None,
            "detail": None if service.is_loaded else service.load_error,
        }
        return JSONResponse(status_code=200 if service.is_loaded else 503, content=body)

    @app.get("/model-info", tags=["info"])
    def model_info(request: Request) -> dict[str, Any]:
        service = require_model(request)
        md = service.metadata
        return {
            "model_type": md.get("model_type"),
            "model_version": service.model_version,
            "threshold": service.threshold,
            "threshold_overridden_by_env": service.threshold_override is not None,
            "score_definition": md.get("score_definition"),
            "feature_schema": md.get("raw_feature_schema", FEATURE_DESCRIPTIONS),
            "model_features": md.get("model_features"),
            "training_metadata": {
                k: md.get(k) for k in (
                    "trained_at_utc", "data_source", "is_synthetic_data", "random_seed", "hyperparameters",
                    "training_rows", "threshold_selection", "sklearn_version", "python_version",
                )
            },
            "evaluation": service.metrics,
            "disclaimer": DISCLAIMER,
        }

    @app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
    def predict(txn: Transaction, request: Request) -> PredictionResponse:
        service = require_model(request)
        results, elapsed_ms = score_and_log(request, service, [txn.model_dump()])
        return PredictionResponse(
            **results[0],
            threshold=service.threshold,
            model_version=service.model_version,
            processing_time_ms=round(elapsed_ms, 3),
            request_id=request.state.request_id,
        )

    @app.post("/predict/batch", response_model=BatchResponse, tags=["prediction"])
    def predict_batch(batch: BatchRequest, request: Request) -> BatchResponse:
        service = require_model(request)
        records = [t.model_dump() for t in batch.transactions]
        results, elapsed_ms = score_and_log(request, service, records)
        return BatchResponse(
            predictions=[PredictionItem(**r) for r in results],
            count=len(results),
            suspicious_count=sum(1 for r in results if r["is_suspicious"]),
            threshold=service.threshold,
            model_version=service.model_version,
            processing_time_ms=round(elapsed_ms, 3),
            request_id=request.state.request_id,
        )

    @app.get("/metrics", tags=["monitoring"])
    def metrics(request: Request) -> dict[str, Any]:
        return request.app.state.monitoring.metrics()

    @app.get("/drift", tags=["monitoring"])
    def drift(request: Request) -> dict[str, Any]:
        return request.app.state.monitoring.drift()

    @app.get("/predictions/recent", tags=["monitoring"])
    def recent_predictions(request: Request, limit: int = Query(200, ge=1, le=1000)) -> dict[str, Any]:
        return {"predictions": request.app.state.db.recent_predictions(limit)}

    @app.post("/demo/generate", response_model=DemoResponse, tags=["demo"])
    def demo_generate(body: DemoRequest, request: Request) -> DemoResponse:
        """Generate SYNTHETIC transactions, score them and log them (demo traffic)."""
        service = require_model(request)
        records = generate_demo_transactions(body.n, seed=body.seed, drift=body.drift)
        results, _ = score_and_log(request, service, records)
        return DemoResponse(
            generated=len(results),
            suspicious=sum(1 for r in results if r["is_suspicious"]),
            drift_simulated=body.drift,
            synthetic=True,
            request_id=request.state.request_id,
        )

    return app
