import json

import numpy as np
import pandas as pd
import pytest

from src.data_generator import generate_demo_transactions, generate_transactions
from src.data_validation import DataValidationError, validate_dataframe
from src.evaluate import metrics_at_threshold, select_threshold_max_f1
from src.features import LABEL_COLUMN, MODEL_FEATURES, RAW_FEATURES, engineer_features
from src.train import QualityGateError, chronological_split, train_pipeline


def test_generator_is_deterministic_and_labelled_synthetic():
    a, b = generate_transactions(1000, 0.01, seed=3), generate_transactions(1000, 0.01, seed=3)
    pd.testing.assert_frame_equal(a, b)
    assert (a["data_source"] == "synthetic").all()
    assert a[LABEL_COLUMN].sum() == 10


def test_demo_transactions_shape():
    rows = generate_demo_transactions(10, seed=1)
    assert len(rows) == 10 and rows[0]["is_synthetic"] is True
    assert set(RAW_FEATURES) <= set(rows[0])


def test_validation_accepts_clean_data():
    clean, report = validate_dataframe(generate_transactions(2000, 0.02, seed=1))
    assert report["n_rows_clean"] == 2000 and report["missing_values"] == 0


def test_validation_rejects_missing_column_and_nan():
    df = generate_transactions(2000, 0.02, seed=1)
    with pytest.raises(DataValidationError):
        validate_dataframe(df.drop(columns=["amount"]))
    bad = df.copy()
    bad.loc[0, "amount"] = np.nan
    with pytest.raises(DataValidationError):
        validate_dataframe(bad)


def test_validation_drops_duplicate_ids_and_rejects_bad_labels():
    df = generate_transactions(2000, 0.02, seed=1)
    dup = pd.concat([df, df.head(5)], ignore_index=True)
    _, report = validate_dataframe(dup)
    assert report["duplicate_transaction_ids_dropped"] == 5
    bad = df.copy()
    bad.loc[0, LABEL_COLUMN] = 7
    with pytest.raises(DataValidationError):
        validate_dataframe(bad)


def test_validation_rejects_out_of_range():
    df = generate_transactions(2000, 0.02, seed=1)
    df.loc[0, "hour"] = 30
    with pytest.raises(DataValidationError):
        validate_dataframe(df)


def test_feature_engineering_columns():
    X = engineer_features(generate_transactions(200, 0.05, seed=1))
    assert list(X.columns) == MODEL_FEATURES
    assert not X.isna().any().any()


def test_chronological_split_has_no_overlap_or_leakage():
    df, _ = validate_dataframe(generate_transactions(2000, 0.02, seed=1))
    train, val, test = chronological_split(df)
    assert len(train) + len(val) + len(test) == len(df)
    assert train["timestamp"].max() <= val["timestamp"].min()
    assert val["timestamp"].max() <= test["timestamp"].min()
    assert not set(train["transaction_id"]) & set(test["transaction_id"])


def test_metrics_and_threshold_selection():
    y = np.array([0] * 95 + [1] * 5)
    scores = np.concatenate([np.linspace(0, 0.5, 95), np.linspace(0.8, 1.0, 5)])
    thr = select_threshold_max_f1(y, scores)
    m = metrics_at_threshold(y, scores, thr)
    assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["pr_auc"] == 1.0
    assert m["confusion_matrix"] == {"tn": 95, "fp": 0, "fn": 0, "tp": 5}
    # "flag nothing" has high accuracy but zero recall
    none = metrics_at_threshold(y, scores, 2.0)
    assert none["accuracy"] == 0.95 and none["recall"] == 0.0


def test_training_artifacts_and_metrics(model_dir):
    for name in ["model.joblib", "baseline_logreg.joblib", "metadata.json", "metrics.json", "reference_stats.json"]:
        assert (model_dir / name).exists()
    metrics = json.loads((model_dir / "metrics.json").read_text())
    test = metrics["isolation_forest"]["test"]
    assert {"precision", "recall", "f1", "pr_auc", "confusion_matrix"} <= set(test)
    assert "not a probability" in metrics["isolation_forest"]["score_type"]


def test_quality_gate_blocks_artifacts(tmp_path):
    with pytest.raises(QualityGateError):
        train_pipeline(out_dir=tmp_path / "out", n_rows=6000, fraud_rate=0.01, seed=7, min_pr_auc=1.01)
    assert not (tmp_path / "out" / "model.joblib").exists()
