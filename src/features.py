"""Feature definitions shared by training, evaluation and the API.

Keeping them in ONE place guarantees that the model sees the same features at
training time and at prediction time (no training/serving skew).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LABEL_COLUMN = "is_fraud"

# Raw fields a client sends to the API (no personal data: no names, card numbers...).
RAW_FEATURES: list[str] = [
    "amount",
    "hour",
    "merchant_category",
    "distance_from_home_km",
    "txn_count_24h",
    "avg_amount_30d",
    "is_online",
    "is_foreign",
    "card_age_days",
]

# Features actually fed to the models after engineering.
MODEL_FEATURES: list[str] = [
    "log_amount",
    "amount_to_avg_ratio",
    "distance_from_home_km",
    "txn_count_24h",
    "is_online",
    "is_foreign",
    "card_age_days",
    "is_night",
    "hour_sin",
    "hour_cos",
    "is_high_risk_merchant",
]

# Raw features compared against the training distribution for drift monitoring.
MONITORED_FEATURES: list[str] = list(RAW_FEATURES)

# Synthetic-data assumption: merchant categories 7-9 carry higher fraud risk.
HIGH_RISK_MERCHANTS: tuple[int, ...] = (7, 8, 9)

FEATURE_DESCRIPTIONS: dict[str, str] = {
    "amount": "Transaction amount (currency units, > 0)",
    "hour": "Hour of day, 0-23",
    "merchant_category": "Merchant category code, 0-9 (synthetic taxonomy)",
    "distance_from_home_km": "Distance between transaction and cardholder home area, km",
    "txn_count_24h": "Number of transactions on the card in the previous 24 hours",
    "avg_amount_30d": "Average transaction amount over the previous 30 days",
    "is_online": "1 if card-not-present / online, else 0",
    "is_foreign": "1 if the merchant is in a foreign country, else 0",
    "card_age_days": "Age of the card in days",
}


def engineer_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn raw transaction fields into the model feature matrix."""
    missing = [c for c in RAW_FEATURES if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing raw feature columns: {missing}")

    df = raw[RAW_FEATURES].astype("float64")
    out = pd.DataFrame(index=raw.index)
    out["log_amount"] = np.log1p(df["amount"])
    out["amount_to_avg_ratio"] = df["amount"] / (df["avg_amount_30d"] + 1.0)
    out["distance_from_home_km"] = df["distance_from_home_km"]
    out["txn_count_24h"] = df["txn_count_24h"]
    out["is_online"] = df["is_online"]
    out["is_foreign"] = df["is_foreign"]
    out["card_age_days"] = df["card_age_days"]
    out["is_night"] = (df["hour"] <= 5).astype("float64")
    angle = 2.0 * np.pi * df["hour"] / 24.0
    out["hour_sin"] = np.sin(angle)
    out["hour_cos"] = np.cos(angle)
    out["is_high_risk_merchant"] = df["merchant_category"].isin(HIGH_RISK_MERCHANTS).astype("float64")
    return out[MODEL_FEATURES]
