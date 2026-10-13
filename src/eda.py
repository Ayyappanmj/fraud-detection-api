"""Lightweight exploratory data analysis (printed + saved as JSON)."""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.features import LABEL_COLUMN, RAW_FEATURES


def run_eda(df: pd.DataFrame) -> dict[str, Any]:
    """Summarise class balance and per-class feature means."""
    counts = df[LABEL_COLUMN].value_counts().to_dict()
    summary: dict[str, Any] = {
        "n_rows": int(len(df)),
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "fraud_rate": float(df[LABEL_COLUMN].mean()),
        "missing_values": int(df[RAW_FEATURES].isna().sum().sum()),
        "duplicate_transaction_ids": int(df["transaction_id"].duplicated().sum()),
        "feature_means_by_class": {
            str(label): {k: round(float(v), 3) for k, v in row.items()}
            for label, row in df.groupby(LABEL_COLUMN)[RAW_FEATURES].mean().iterrows()
        },
        "feature_describe": {
            col: {k: round(float(v), 3) for k, v in df[col].describe().items()} for col in RAW_FEATURES
        },
    }
    return summary


def print_eda(summary: dict[str, Any]) -> None:
    print("=== EDA ===")
    print(f"rows: {summary['n_rows']:,}  fraud rate: {summary['fraud_rate']:.3%}")
    print(f"class counts: {summary['class_counts']}")
    print(f"missing values: {summary['missing_values']}  duplicate ids: {summary['duplicate_transaction_ids']}")
    means = pd.DataFrame(summary["feature_means_by_class"])
    means.columns = [f"class_{c}" for c in means.columns]
    print("feature means by class (0 = normal, 1 = fraud):")
    print(means.to_string())
