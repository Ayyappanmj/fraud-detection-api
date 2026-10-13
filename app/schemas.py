"""Pydantic request/response models (input validation lives here)."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

_EXAMPLE_TXN = {
    "transaction_id": "demo-0001",
    "amount": 245.5,
    "hour": 2,
    "merchant_category": 8,
    "distance_from_home_km": 320.0,
    "txn_count_24h": 7,
    "avg_amount_30d": 38.2,
    "is_online": 1,
    "is_foreign": 1,
    "card_age_days": 20,
    "is_synthetic": True,
}


class Transaction(BaseModel):
    """One transaction. Contains NO personal data (no names or card numbers)."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": _EXAMPLE_TXN})

    transaction_id: Optional[str] = Field(
        default=None, max_length=64, pattern=r"^[A-Za-z0-9_\-]+$",
        description="Optional client id (letters, digits, - and _). Generated if omitted.",
    )
    amount: float = Field(..., gt=0, le=1_000_000, allow_inf_nan=False)
    hour: int = Field(..., ge=0, le=23)
    merchant_category: int = Field(..., ge=0, le=9)
    distance_from_home_km: float = Field(..., ge=0, le=20_000, allow_inf_nan=False)
    txn_count_24h: int = Field(..., ge=0, le=1000)
    avg_amount_30d: float = Field(..., ge=0, le=1_000_000, allow_inf_nan=False)
    is_online: int = Field(..., ge=0, le=1)
    is_foreign: int = Field(..., ge=0, le=1)
    card_age_days: int = Field(..., ge=0, le=36_500)
    is_synthetic: bool = Field(default=False, description="Mark demo/synthetic traffic")


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transactions: list[Transaction] = Field(..., min_length=1, max_length=1000)


class PredictionItem(BaseModel):
    transaction_id: str
    anomaly_score: float = Field(description="Isolation Forest anomaly score. NOT a fraud probability.")
    classification: Literal["suspicious", "normal"]
    is_suspicious: bool


class PredictionResponse(PredictionItem):
    model_config = ConfigDict(protected_namespaces=())
    threshold: float
    model_version: str
    processing_time_ms: float
    request_id: str


class BatchResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    predictions: list[PredictionItem]
    count: int
    suspicious_count: int
    threshold: float
    model_version: str
    processing_time_ms: float
    request_id: str


class DemoRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n: int = Field(default=50, ge=1, le=500)
    drift: bool = Field(default=False, description="Shift the input distribution to demo drift detection")
    seed: Optional[int] = Field(default=None, ge=0, le=2**31 - 1)


class DemoResponse(BaseModel):
    generated: int
    suspicious: int
    drift_simulated: bool
    synthetic: bool
    request_id: str
