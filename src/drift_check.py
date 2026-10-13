"""Lightweight data-drift monitoring (Population Stability Index + KS test).

This compares the distribution of INCOMING INPUT FEATURES with the training
reference. It does NOT measure fraud rate, label drift or model accuracy, and a
drift warning does not mean fraud is happening or that the model has failed.

Usage (simulation on synthetic data):
    python -m src.drift_check --model-dir models --n 500 --drift
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.features import MONITORED_FEATURES
from src.io_utils import json_default, read_json

EPS = 1e-4
DEFAULT_PSI_THRESHOLD = 0.2   # rule of thumb: <0.1 stable, 0.1-0.2 moderate, >0.2 significant
DEFAULT_MIN_SAMPLES = 100     # below this, PSI/KS on 10 bins is too noisy to be meaningful

DRIFT_DISCLAIMER = (
    "Monitors input-feature distribution changes only. Drift does not mean fraud and does not "
    "prove model failure; changing fraud behaviour may not appear as input drift."
)


def _fmt(x: float) -> str:
    return f"{x:.4g}"


def _bin_spec(values: np.ndarray, max_categories: int = 10, n_bins: int = 10) -> tuple[str, list[float], list[str]]:
    """Return (kind, inner_edges, labels). Few unique values => one bin per value."""
    uniques = np.unique(values)
    if len(uniques) <= max_categories:
        inner = ((uniques[:-1] + uniques[1:]) / 2.0).tolist()
        return "categorical", inner, [_fmt(u) for u in uniques]
    inner = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)[1:-1])).tolist()
    labels = [f"<= {_fmt(inner[0])}"]
    labels += [f"{_fmt(inner[i - 1])} to {_fmt(inner[i])}" for i in range(1, len(inner))]
    labels.append(f"> {_fmt(inner[-1])}")
    return "numeric", inner, labels


def _proportions(values: np.ndarray, inner_edges: list[float]) -> np.ndarray:
    idx = np.searchsorted(np.asarray(inner_edges), values, side="right")
    counts = np.bincount(idx, minlength=len(inner_edges) + 1)
    return counts / max(len(values), 1)


def compute_psi(reference_props: np.ndarray, current_props: np.ndarray) -> float:
    """Population Stability Index between two binned distributions."""
    p = np.clip(np.asarray(reference_props, dtype=float), EPS, None)
    q = np.clip(np.asarray(current_props, dtype=float), EPS, None)
    return float(np.sum((q - p) * np.log(q / p)))


def build_reference_stats(
    df: pd.DataFrame,
    features: list[str] = MONITORED_FEATURES,
    seed: int = 42,
    sample_size: int = 1000,
) -> dict[str, Any]:
    """Summarise the TRAINING data so the API can compare live data against it."""
    rng = np.random.default_rng(seed)
    stats: dict[str, Any] = {}
    for name in features:
        values = df[name].to_numpy(dtype=float)
        kind, inner, labels = _bin_spec(values)
        sample = rng.choice(values, size=min(sample_size, len(values)), replace=False)
        stats[name] = {
            "kind": kind,
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "p05": float(np.quantile(values, 0.05)),
            "p50": float(np.quantile(values, 0.50)),
            "p95": float(np.quantile(values, 0.95)),
            "bin_edges": inner,
            "bin_labels": labels,
            "bin_proportions": _proportions(values, inner).tolist(),
            "sample": np.round(sample, 4).tolist(),
        }
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": int(len(df)),
        "features": stats,
    }


def compute_drift(
    reference: dict[str, Any],
    recent: pd.DataFrame,
    psi_threshold: float = DEFAULT_PSI_THRESHOLD,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> dict[str, Any]:
    """Compare ``recent`` rows with ``reference`` and return a JSON-safe report."""
    n_recent = int(len(recent))
    enough = n_recent >= min_samples
    report: dict[str, Any] = {
        "status": "insufficient_data",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "n_recent": n_recent,
        "min_samples": int(min_samples),
        "psi_threshold": float(psi_threshold),
        "drifted_features": [],
        "features": {},
        "message": "",
        "disclaimer": DRIFT_DISCLAIMER,
    }

    for name, ref in reference["features"].items():
        if n_recent == 0 or name not in recent.columns:
            continue
        values = recent[name].to_numpy(dtype=float)
        recent_props = _proportions(values, ref["bin_edges"])
        entry: dict[str, Any] = {
            "reference": {k: ref[k] for k in ("mean", "std", "min", "max", "p05", "p50", "p95")},
            "recent": {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
                "p05": float(np.quantile(values, 0.05)),
                "p50": float(np.quantile(values, 0.50)),
                "p95": float(np.quantile(values, 0.95)),
            },
            "bin_labels": ref["bin_labels"],
            "reference_hist": ref["bin_proportions"],
            "recent_hist": recent_props.tolist(),
            "psi": None,
            "ks_statistic": None,
            "ks_pvalue": None,
            "drifted": False,
        }
        if enough:
            entry["psi"] = compute_psi(np.asarray(ref["bin_proportions"]), recent_props)
            ks = ks_2samp(np.asarray(ref["sample"], dtype=float), values)
            entry["ks_statistic"] = float(ks.statistic)
            entry["ks_pvalue"] = float(ks.pvalue)
            entry["drifted"] = bool(entry["psi"] >= psi_threshold)
            if entry["drifted"]:
                report["drifted_features"].append(name)
        report["features"][name] = entry

    if not enough:
        report["message"] = (
            f"Only {n_recent} recent transactions; at least {min_samples} are needed before drift "
            "warnings are meaningful."
        )
    elif report["drifted_features"]:
        report["status"] = "drift_warning"
        report["message"] = (
            f"{len(report['drifted_features'])} feature(s) exceed PSI {psi_threshold}: "
            f"{', '.join(report['drifted_features'])}. Investigate; this is not proof of fraud or model failure."
        )
    else:
        report["status"] = "no_drift"
        report["message"] = "No feature exceeds the configured PSI threshold."
    return report


def main() -> None:
    from src.data_generator import generate_demo_transactions

    parser = argparse.ArgumentParser(description="Simulate a drift check on synthetic data.")
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--drift", action="store_true", help="Shift the synthetic input distribution")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    reference = read_json(args.model_dir / "reference_stats.json")
    recent = pd.DataFrame(generate_demo_transactions(args.n, seed=args.seed, drift=args.drift))
    report = compute_drift(reference, recent)
    slim = {k: v for k, v in report.items() if k != "features"}
    slim["psi_by_feature"] = {k: (None if v["psi"] is None else round(v["psi"], 3)) for k, v in report["features"].items()}
    print(json.dumps(slim, indent=2, default=json_default))


if __name__ == "__main__":
    main()
