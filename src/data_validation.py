"""Dataset validation: schema, missing values, ranges, duplicates, labels.

Usage:
    python -m src.data_validation --data-path data/synthetic_transactions.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features import LABEL_COLUMN, RAW_FEATURES

REQUIRED_COLUMNS: list[str] = ["transaction_id", "timestamp", *RAW_FEATURES]
VALID_RANGES: dict[str, tuple[float, float]] = {
    "amount": (0.0, 1_000_000.0),
    "hour": (0, 23),
    "merchant_category": (0, 9),
    "distance_from_home_km": (0.0, 20_000.0),
    "txn_count_24h": (0, 1000),
    "avg_amount_30d": (0.0, 1_000_000.0),
    "is_online": (0, 1),
    "is_foreign": (0, 1),
    "card_age_days": (0, 36_500),
}
MIN_ROWS = 500
MIN_POSITIVES_TOTAL = 30


class DataValidationError(ValueError):
    """Raised when a dataset is not safe to train on."""


def validate_dataframe(
    df: pd.DataFrame, require_label: bool = True
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate ``df`` and return ``(clean_df, report)``.

    Raises ``DataValidationError`` for problems that make training unsafe.
    Duplicate transaction ids are dropped (and reported), never silently kept.
    """
    required = REQUIRED_COLUMNS + ([LABEL_COLUMN] if require_label else [])
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise DataValidationError(f"Missing required columns: {missing_cols}")

    df = df.copy()
    if len(df) < MIN_ROWS:
        raise DataValidationError(f"Need at least {MIN_ROWS} rows, got {len(df)}")

    n_missing = int(df[required].isna().sum().sum())
    if n_missing:
        raise DataValidationError(f"Found {n_missing} missing values in required columns")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if df["timestamp"].isna().any():
        raise DataValidationError("Some timestamps could not be parsed")

    for col in RAW_FEATURES:
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise DataValidationError(f"Column '{col}' has non-numeric or non-finite values")
        low, high = VALID_RANGES[col]
        if (numeric < low).any() or (numeric > high).any():
            raise DataValidationError(f"Column '{col}' has values outside [{low}, {high}]")
        df[col] = numeric

    n_dup_ids = int(df["transaction_id"].duplicated().sum())
    n_dup_rows_ignoring_id = int(df.duplicated(subset=["timestamp", *RAW_FEATURES]).sum())
    df = df.drop_duplicates(subset="transaction_id", keep="first")

    report: dict[str, Any] = {
        "n_rows_input": int(len(df) + n_dup_ids),
        "n_rows_clean": int(len(df)),
        "missing_values": n_missing,
        "duplicate_transaction_ids_dropped": n_dup_ids,
        "duplicate_rows_ignoring_id": n_dup_rows_ignoring_id,
        "time_range": [df["timestamp"].min().isoformat(), df["timestamp"].max().isoformat()],
    }

    if require_label:
        labels = pd.to_numeric(df[LABEL_COLUMN], errors="coerce")
        if not labels.isin([0, 1]).all():
            raise DataValidationError(f"'{LABEL_COLUMN}' must contain only 0 and 1")
        df[LABEL_COLUMN] = labels.astype(int)
        positives = int(df[LABEL_COLUMN].sum())
        if positives < MIN_POSITIVES_TOTAL:
            raise DataValidationError(
                f"Only {positives} fraud rows; need at least {MIN_POSITIVES_TOTAL} to train and evaluate"
            )
        report["n_fraud"] = positives
        report["fraud_rate"] = float(positives / len(df))

    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    return df, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a transaction CSV.")
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--no-label", action="store_true", help="Do not require the label column")
    args = parser.parse_args()
    try:
        _, report = validate_dataframe(pd.read_csv(args.data_path), require_label=not args.no_label)
    except (DataValidationError, FileNotFoundError) as exc:
        print(f"VALIDATION FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
    print("VALIDATION PASSED")
    for key, value in report.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
