"""Evaluation helpers for imbalanced fraud detection.

Accuracy alone is misleading when only ~0.5% of rows are fraud: a model that
flags nothing is ~99.5% accurate and completely useless. We therefore report
precision, recall, F1, PR-AUC and the confusion matrix.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data_generator import generate_transactions
from src.data_validation import validate_dataframe
from src.features import LABEL_COLUMN, engineer_features
from src.io_utils import json_default, read_json


def metrics_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    """Compute classification metrics when flagging ``score >= threshold``."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(scores) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    positives = int(y_true.sum())
    has_both = positives > 0 and positives < len(y_true)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, scores)) if positives else None,
        "roc_auc": float(roc_auc_score(y_true, scores)) if has_both else None,
        "accuracy": float((tp + tn) / len(y_true)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "alerts": int(tp + fp),
        "alert_rate": float((tp + fp) / len(y_true)),
        "n": int(len(y_true)),
        "positives": positives,
        "prevalence": float(positives / len(y_true)),
    }


def select_threshold_max_f1(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Pick the threshold that maximises F1. Use VALIDATION data only (never test)."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    denom = precision[:-1] + recall[:-1]
    f1 = np.divide(2 * precision[:-1] * recall[:-1], denom, out=np.zeros_like(denom), where=denom > 0)
    if len(f1) == 0 or f1.max() <= 0:
        return float(np.quantile(scores, 0.99))
    return float(thresholds[int(np.argmax(f1))])


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the saved model on freshly generated synthetic data.")
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--n-rows", type=int, default=20_000)
    parser.add_argument("--fraud-rate", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=123, help="Use a seed different from training")
    args = parser.parse_args()

    bundle = joblib.load(args.model_dir / "model.joblib")
    metadata = read_json(args.model_dir / "metadata.json")
    df, _ = validate_dataframe(generate_transactions(args.n_rows, args.fraud_rate, args.seed))
    scores = -bundle["model"].score_samples(engineer_features(df))
    result = metrics_at_threshold(df[LABEL_COLUMN].to_numpy(), scores, metadata["threshold"])
    print(json.dumps(result, indent=2, default=json_default))
    print("NOTE: fresh SYNTHETIC data from the same generator; not proof of real-world performance.")


if __name__ == "__main__":
    main()
