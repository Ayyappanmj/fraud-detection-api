import numpy as np
import pandas as pd

from src.data_generator import generate_demo_transactions, generate_transactions
from src.drift_check import build_reference_stats, compute_drift, compute_psi
from src.features import RAW_FEATURES


def _reference():
    return build_reference_stats(generate_transactions(5000, 0.01, seed=1))


def _recent(n, seed, drift=False):
    return pd.DataFrame(generate_demo_transactions(n, seed=seed, drift=drift))[RAW_FEATURES]


def test_psi_zero_for_identical_distributions():
    p = np.array([0.2, 0.3, 0.5])
    assert compute_psi(p, p) == 0.0


def test_psi_positive_for_shifted_distributions():
    assert compute_psi(np.array([0.5, 0.5]), np.array([0.9, 0.1])) > 0.2


def test_insufficient_data_status():
    report = compute_drift(_reference(), _recent(50, 2), min_samples=100)
    assert report["status"] == "insufficient_data"
    assert report["drifted_features"] == []
    assert all(f["psi"] is None for f in report["features"].values())


def test_empty_recent_data():
    report = compute_drift(_reference(), pd.DataFrame(columns=RAW_FEATURES))
    assert report["status"] == "insufficient_data" and report["n_recent"] == 0


def test_same_distribution_no_drift():
    report = compute_drift(_reference(), _recent(500, 3))
    assert report["status"] == "no_drift"


def test_shifted_distribution_triggers_warning():
    report = compute_drift(_reference(), _recent(500, 4, drift=True))
    assert report["status"] == "drift_warning"
    assert {"amount", "txn_count_24h"} <= set(report["drifted_features"])
    assert report["features"]["amount"]["psi"] > report["psi_threshold"]


def test_threshold_is_configurable():
    ref, recent = _reference(), _recent(500, 5, drift=True)
    assert compute_drift(ref, recent, psi_threshold=1000.0)["status"] == "no_drift"


def test_report_is_json_serialisable_and_has_disclaimer():
    import json
    report = compute_drift(_reference(), _recent(300, 6))
    json.dumps(report)
    assert "does not mean fraud" in report["disclaimer"]
