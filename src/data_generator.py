"""Synthetic credit-card-style transaction generator.

ALL data produced here is SYNTHETIC. It contains no real cardholder data and
is only meant for development, testing and demonstrations.

Usage:
    python -m src.data_generator --n-rows 40000 --fraud-rate 0.005 \
        --seed 42 --output data/synthetic_transactions.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features import LABEL_COLUMN, RAW_FEATURES

_NORMAL_HOUR_WEIGHTS = np.array(
    [1, 1, 1, 1, 1, 2, 4, 6, 8, 8, 8, 8, 9, 9, 8, 8, 8, 9, 9, 8, 6, 4, 3, 2], dtype=float
)
_NORMAL_MERCHANT_P = np.array([0.18, 0.16, 0.14, 0.12, 0.10, 0.10, 0.08, 0.04, 0.04, 0.04])


def _normal_features(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    return {
        "amount": rng.lognormal(3.4, 0.8, n),
        "hour": rng.choice(24, size=n, p=_NORMAL_HOUR_WEIGHTS / _NORMAL_HOUR_WEIGHTS.sum()),
        "merchant_category": rng.choice(10, size=n, p=_NORMAL_MERCHANT_P),
        "distance_from_home_km": rng.exponential(10.0, n),
        "txn_count_24h": rng.poisson(2.0, n),
        "avg_amount_30d": rng.lognormal(3.5, 0.4, n),
        "is_online": (rng.random(n) < 0.30).astype(int),
        "is_foreign": (rng.random(n) < 0.03).astype(int),
        "card_age_days": rng.integers(30, 3650, n),
    }


def _apply_fraud(
    rng: np.random.Generator, f: dict[str, np.ndarray], idx: np.ndarray, severity: np.ndarray
) -> None:
    """Overwrite rows ``idx`` with fraud-like behaviour.

    ``severity`` is 1.0 for obvious fraud and 0.5 for "subtle" fraud that
    overlaps with normal behaviour (so the task is not trivially easy).
    """
    k = len(idx)
    w = severity
    f["amount"][idx] = rng.lognormal(3.4 + 1.8 * w, 0.9)
    f["distance_from_home_km"][idx] = rng.exponential(10.0 + 120.0 * w)
    f["txn_count_24h"][idx] = rng.poisson(2.0 + 5.0 * w)
    f["is_online"][idx] = (rng.random(k) < 0.30 + 0.5 * w).astype(int)
    f["is_foreign"][idx] = (rng.random(k) < 0.03 + 0.35 * w).astype(int)
    night = rng.random(k) < 0.6 * w
    f["hour"][idx] = np.where(night, rng.integers(0, 6, k), f["hour"][idx])
    risky = rng.random(k) < 0.7 * w
    f["merchant_category"][idx] = np.where(risky, rng.integers(7, 10, k), f["merchant_category"][idx])
    new_card = rng.random(k) < 0.4 * w
    f["card_age_days"][idx] = np.where(new_card, rng.integers(1, 60, k), f["card_age_days"][idx])


def generate_transactions(
    n_rows: int = 40_000,
    fraud_rate: float = 0.005,
    seed: int = 42,
    start: str = "2024-01-01",
    days: int = 60,
) -> pd.DataFrame:
    """Return a labelled synthetic dataset sorted by timestamp."""
    if n_rows < 100:
        raise ValueError("n_rows must be at least 100")
    if not 0.0 < fraud_rate < 0.5:
        raise ValueError("fraud_rate must be between 0 and 0.5 (exclusive)")

    rng = np.random.default_rng(seed)
    feats = _normal_features(rng, n_rows)

    n_fraud = max(1, int(round(n_rows * fraud_rate)))
    fraud_idx = rng.choice(n_rows, size=n_fraud, replace=False)
    severity = np.where(rng.random(n_fraud) < 0.35, 0.5, 1.0)
    _apply_fraud(rng, feats, fraud_idx, severity)

    feats["amount"] = np.round(np.clip(feats["amount"], 0.5, 50_000), 2)
    feats["avg_amount_30d"] = np.round(np.clip(feats["avg_amount_30d"], 0.5, 50_000), 2)
    feats["distance_from_home_km"] = np.round(np.clip(feats["distance_from_home_km"], 0, 20_000), 1)
    feats["txn_count_24h"] = np.clip(feats["txn_count_24h"], 0, 1000)

    labels = np.zeros(n_rows, dtype=int)
    labels[fraud_idx] = 1

    offsets = np.sort(rng.uniform(0, days * 86_400, n_rows))
    timestamps = (pd.Timestamp(start, tz="UTC") + pd.to_timedelta(offsets, unit="s")).floor("s")

    df = pd.DataFrame(feats)
    df.insert(0, "timestamp", timestamps)
    df.insert(0, "transaction_id", [f"SYN-{i:07d}" for i in range(n_rows)])
    df[LABEL_COLUMN] = labels
    df["data_source"] = "synthetic"
    return df[["transaction_id", "timestamp", *RAW_FEATURES, LABEL_COLUMN, "data_source"]]


def generate_demo_transactions(
    n: int = 50, seed: int | None = None, drift: bool = False, fraud_rate: float = 0.05
) -> list[dict[str, Any]]:
    """Unlabelled synthetic transactions for demos (API ``/demo/generate``).

    With ``drift=True`` the input distribution is deliberately shifted so you
    can see the drift monitor react.
    """
    if seed is None:
        seed = int(np.random.default_rng().integers(0, 2**31 - 1))
    df = generate_transactions(max(n, 100), fraud_rate=fraud_rate, seed=seed).head(n).copy()
    if drift:
        rng = np.random.default_rng(seed + 1)
        df["amount"] = (df["amount"] * 3).clip(upper=50_000).round(2)
        df["distance_from_home_km"] = (df["distance_from_home_km"] * 2.5).clip(upper=20_000).round(1)
        df["txn_count_24h"] = df["txn_count_24h"] + 3
        df["is_online"] = (rng.random(len(df)) < 0.75).astype(int)
    df["transaction_id"] = [f"DEMO-{seed}-{i:04d}" for i in range(len(df))]
    df["is_synthetic"] = True
    cols = ["transaction_id", *RAW_FEATURES, "is_synthetic"]
    return json.loads(df[cols].to_json(orient="records"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic transaction dataset (CSV).")
    parser.add_argument("--n-rows", type=int, default=40_000)
    parser.add_argument("--fraud-rate", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/synthetic_transactions.csv"))
    args = parser.parse_args()

    df = generate_transactions(args.n_rows, args.fraud_rate, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df):,} SYNTHETIC rows to {args.output}")
    print(f"Fraud rows: {int(df[LABEL_COLUMN].sum())} ({df[LABEL_COLUMN].mean():.3%})")


if __name__ == "__main__":
    main()
