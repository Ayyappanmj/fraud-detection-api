"""Reproducible training pipeline.

    python -m src.train                                  # synthetic data, defaults
    python -m src.train --data-path data/my_data.csv     # your own labelled CSV

Steps: load/generate -> validate -> EDA -> engineer features -> chronological
train/val/test split -> fit models on TRAIN only -> choose threshold on
VALIDATION only -> evaluate once on TEST -> quality gate -> save artifacts.

Isolation Forest is unsupervised. Its output is an ANOMALY SCORE (higher = more
unusual); it is NOT a calibrated probability of fraud.
"""
from __future__ import annotations

import argparse
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_generator import generate_transactions
from src.data_validation import DataValidationError, validate_dataframe
from src.drift_check import build_reference_stats
from src.eda import print_eda, run_eda
from src.evaluate import metrics_at_threshold, select_threshold_max_f1
from src.features import (
    FEATURE_DESCRIPTIONS,
    LABEL_COLUMN,
    MODEL_FEATURES,
    MONITORED_FEATURES,
    RAW_FEATURES,
    engineer_features,
)
from src.io_utils import sha256_file, write_json

MIN_POSITIVES_PER_SPLIT = 5
MODEL_BASE_VERSION = "1.0.0"

SCORE_NOTE = (
    "anomaly_score = -IsolationForest.score_samples(X). Higher means more unusual. It is a relative "
    "anomaly score, NOT a calibrated probability of fraud."
)


class QualityGateError(RuntimeError):
    """Raised when the new model fails the evaluation gate (artifacts are NOT written)."""


def chronological_split(df: pd.DataFrame, train_frac: float = 0.6, val_frac: float = 0.2):
    """Split a time-sorted frame into past (train) -> validation -> future (test).

    Chronological splitting avoids leaking future information into training.
    """
    n = len(df)
    i1, i2 = int(n * train_frac), int(n * (train_frac + val_frac))
    return df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]


def train_pipeline(
    out_dir: Path,
    data_path: Path | None = None,
    n_rows: int = 40_000,
    fraud_rate: float = 0.005,
    seed: int = 42,
    min_pr_auc: float = 0.10,
    verbose: bool = False,
) -> dict[str, Any]:
    out_dir = Path(out_dir)

    # 1-2. Load + validate
    if data_path is not None:
        raw = pd.read_csv(data_path)
        data_source = f"csv:{Path(data_path).name}"
    else:
        raw = generate_transactions(n_rows=n_rows, fraud_rate=fraud_rate, seed=seed)
        data_source = "synthetic_generator"
    df, validation_report = validate_dataframe(raw, require_label=True)

    # 3. EDA
    eda = run_eda(df)
    if verbose:
        print_eda(eda)

    # 4-5. Split (before fitting anything => no leakage), then features
    train_df, val_df, test_df = chronological_split(df)
    for name, part in (("train", train_df), ("validation", val_df), ("test", test_df)):
        if int(part[LABEL_COLUMN].sum()) < MIN_POSITIVES_PER_SPLIT:
            raise DataValidationError(f"{name} split has fewer than {MIN_POSITIVES_PER_SPLIT} fraud rows")
    x_train, x_val, x_test = (engineer_features(p) for p in (train_df, val_df, test_df))
    y_train, y_val, y_test = (p[LABEL_COLUMN].to_numpy() for p in (train_df, val_df, test_df))

    # 6. Isolation Forest (unsupervised; labels are NOT used to fit it).
    # Trees do not need feature scaling, so none is applied here.
    iso = IsolationForest(n_estimators=200, max_samples=256, contamination="auto", random_state=seed, n_jobs=1)
    iso.fit(x_train)
    val_scores = -iso.score_samples(x_val)
    test_scores = -iso.score_samples(x_test)

    # 7. Threshold from validation only
    threshold = select_threshold_max_f1(y_val, val_scores)
    if_val = metrics_at_threshold(y_val, val_scores, threshold)
    if_test = metrics_at_threshold(y_test, test_scores, threshold)

    # 8. Supervised baseline: scaling matters for logistic regression.
    logreg = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)),
    ])
    logreg.fit(x_train, y_train)
    lr_val_scores = logreg.predict_proba(x_val)[:, 1]
    lr_test_scores = logreg.predict_proba(x_test)[:, 1]
    lr_threshold = select_threshold_max_f1(y_val, lr_val_scores)
    lr_val = metrics_at_threshold(y_val, lr_val_scores, lr_threshold)
    lr_test = metrics_at_threshold(y_test, lr_test_scores, lr_threshold)

    # 9. Quality gate (before anything is written to disk)
    test_pr_auc = if_test["pr_auc"] or 0.0
    if test_pr_auc < min_pr_auc:
        raise QualityGateError(f"Test PR-AUC {test_pr_auc:.3f} is below the gate {min_pr_auc:.3f}")

    metrics = {
        "dataset": {
            **validation_report,
            "split": "chronological 60/20/20 (train/validation/test)",
            "n_train": len(train_df), "n_validation": len(val_df), "n_test": len(test_df),
            "fraud_train": int(y_train.sum()), "fraud_validation": int(y_val.sum()), "fraud_test": int(y_test.sum()),
        },
        "isolation_forest": {
            "score_type": "anomaly_score (not a probability)",
            "threshold": threshold,
            "threshold_selection": "maximises F1 on the validation split",
            "validation": if_val,
            "test": if_test,
        },
        "logistic_regression": {
            "score_type": "predict_proba from a class_weight='balanced' model (not calibrated)",
            "threshold": lr_threshold,
            "threshold_selection": "maximises F1 on the validation split",
            "validation": lr_val,
            "test": lr_test,
        },
        "baseline_flag_nothing": {"test_accuracy": float(1.0 - y_test.mean())},
        "notes": [
            "Accuracy is misleading under severe class imbalance; prefer PR-AUC, precision and recall.",
            "Results come from SYNTHETIC data unless you supplied your own CSV.",
        ],
    }

    # 10. Save artifacts
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.joblib"
    joblib.dump({"model": iso, "features": MODEL_FEATURES}, model_path)
    joblib.dump(logreg, out_dir / "baseline_logreg.joblib")
    sha = sha256_file(model_path)

    metadata = {
        "model_type": "IsolationForest",
        "model_version": f"{MODEL_BASE_VERSION}-{sha[:8]}",
        "model_sha256": sha,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_source": data_source,
        "is_synthetic_data": data_source == "synthetic_generator",
        "random_seed": seed,
        "hyperparameters": {"n_estimators": 200, "max_samples": 256, "contamination": "auto"},
        "threshold": threshold,
        "threshold_selection": "maximises F1 on the validation split",
        "score_definition": SCORE_NOTE,
        "raw_feature_schema": {c: FEATURE_DESCRIPTIONS[c] for c in RAW_FEATURES},
        "model_features": MODEL_FEATURES,
        "monitored_features": MONITORED_FEATURES,
        "training_rows": len(train_df),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
    }
    write_json(out_dir / "metadata.json", metadata)
    write_json(out_dir / "metrics.json", metrics)
    write_json(out_dir / "eda_summary.json", eda)
    write_json(out_dir / "reference_stats.json", build_reference_stats(train_df, seed=seed))

    return {"metadata": metadata, "metrics": metrics}


def _print_summary(result: dict[str, Any], out_dir: Path) -> None:
    m = result["metrics"]
    print("\n=== RESULTS (test split, threshold chosen on validation) ===")
    for key, title in (("isolation_forest", "Isolation Forest"), ("logistic_regression", "Logistic Regression (supervised baseline)")):
        t = m[key]["test"]
        cm = t["confusion_matrix"]
        print(f"\n{title}")
        print(f"  precision={t['precision']:.3f} recall={t['recall']:.3f} f1={t['f1']:.3f} "
              f"PR-AUC={t['pr_auc']:.3f} ROC-AUC={t['roc_auc']:.3f}")
        print(f"  confusion: TN={cm['tn']} FP={cm['fp']} FN={cm['fn']} TP={cm['tp']}  accuracy={t['accuracy']:.4f}")
    print(f"\n'Flag nothing' accuracy on test: {m['baseline_flag_nothing']['test_accuracy']:.4f} "
          "(why accuracy alone is misleading)")
    print(f"\nArtifacts written to {out_dir.resolve()}")
    print(f"Model version: {result['metadata']['model_version']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the fraud anomaly-detection model.")
    parser.add_argument("--data-path", type=Path, default=None, help="Labelled CSV; default = synthetic generator")
    parser.add_argument("--n-rows", type=int, default=40_000)
    parser.add_argument("--fraud-rate", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("models"))
    parser.add_argument("--min-pr-auc", type=float, default=0.10, help="Quality gate on TEST PR-AUC")
    args = parser.parse_args()
    try:
        result = train_pipeline(args.out_dir, args.data_path, args.n_rows, args.fraud_rate, args.seed,
                                args.min_pr_auc, verbose=True)
    except (DataValidationError, QualityGateError, FileNotFoundError) as exc:
        print(f"TRAINING STOPPED: {exc}", file=sys.stderr)
        sys.exit(1)
    _print_summary(result, args.out_dir)


if __name__ == "__main__":
    main()
