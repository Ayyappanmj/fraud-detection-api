"""Shared fixtures. Tests train a SMALL model into a temp dir, so they never
depend on (or modify) the artifacts in ./models."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from src.train import train_pipeline

VALID_TXN = {
    "transaction_id": "test-001",
    "amount": 25.0,
    "hour": 14,
    "merchant_category": 2,
    "distance_from_home_km": 5.0,
    "txn_count_24h": 2,
    "avg_amount_30d": 30.0,
    "is_online": 0,
    "is_foreign": 0,
    "card_age_days": 900,
}

SUSPICIOUS_TXN = {
    "transaction_id": "test-sus",
    "amount": 4000.0,
    "hour": 3,
    "merchant_category": 9,
    "distance_from_home_km": 2500.0,
    "txn_count_24h": 15,
    "avg_amount_30d": 30.0,
    "is_online": 1,
    "is_foreign": 1,
    "card_age_days": 5,
}


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("models")
    train_pipeline(out_dir=out, n_rows=6000, fraud_rate=0.01, seed=7, min_pr_auc=0.0)
    return out


@pytest.fixture()
def app(model_dir, tmp_path):
    return create_app(model_dir=model_dir, db_path=tmp_path / "test.db")


@pytest.fixture()
def client(app):
    return TestClient(app)


@pytest.fixture()
def client_no_model(tmp_path):
    empty = tmp_path / "empty_models"
    empty.mkdir()
    return TestClient(create_app(model_dir=empty, db_path=tmp_path / "nomodel.db"))
