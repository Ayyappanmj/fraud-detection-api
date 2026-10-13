"""SQLite logging of requests and predictions.

SQLite is a local file. On Render's free tier the filesystem is EPHEMERAL: the
file is wiped on restart/redeploy and after the service sleeps and is rebuilt.
The API works without any history; the dashboard just shows empty charts.
"""
from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.features import RAW_FEATURES

_PRED_COLS = [
    "request_id", "transaction_id", "created_at", "is_synthetic", "anomaly_score",
    "is_suspicious", "threshold", "model_version", "processing_ms", *RAW_FEATURES,
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def init(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        feature_cols = ", ".join(f"{c} REAL" for c in RAW_FEATURES)
        with self._lock:
            conn = self._connect()
            try:
                with conn:
                    conn.execute(
                        "CREATE TABLE IF NOT EXISTS predictions ("
                        "id INTEGER PRIMARY KEY AUTOINCREMENT, request_id TEXT, transaction_id TEXT, "
                        "created_at TEXT, is_synthetic INTEGER, anomaly_score REAL, is_suspicious INTEGER, "
                        f"threshold REAL, model_version TEXT, processing_ms REAL, {feature_cols})"
                    )
                    conn.execute(
                        "CREATE TABLE IF NOT EXISTS requests ("
                        "id INTEGER PRIMARY KEY AUTOINCREMENT, request_id TEXT, created_at TEXT, "
                        "method TEXT, path TEXT, status_code INTEGER, latency_ms REAL)"
                    )
            finally:
                conn.close()

    def _write(self, sql: str, rows: list[tuple]) -> None:
        with self._lock:
            conn = self._connect()
            try:
                with conn:
                    conn.executemany(sql, rows)
            finally:
                conn.close()

    def _read(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        conn = self._connect()
        try:
            return conn.execute(sql, params).fetchall()
        finally:
            conn.close()

    # ---- writes -------------------------------------------------------
    def log_request(self, request_id: str, method: str, path: str, status_code: int, latency_ms: float) -> None:
        self._write(
            "INSERT INTO requests (request_id, created_at, method, path, status_code, latency_ms) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [(request_id, _now(), method, path, int(status_code), float(latency_ms))],
        )

    def log_predictions(
        self,
        request_id: str,
        records: list[dict[str, Any]],
        results: list[dict[str, Any]],
        threshold: float,
        model_version: str,
        total_ms: float,
    ) -> None:
        per_row_ms = float(total_ms) / max(len(results), 1)
        now = _now()
        rows = []
        for rec, res in zip(records, results):
            rows.append((
                request_id, str(res["transaction_id"]), now, int(bool(rec.get("is_synthetic", False))),
                float(res["anomaly_score"]), int(bool(res["is_suspicious"])), float(threshold),
                model_version, per_row_ms, *[float(rec[c]) for c in RAW_FEATURES],
            ))
        placeholders = ", ".join("?" for _ in _PRED_COLS)
        self._write(f"INSERT INTO predictions ({', '.join(_PRED_COLS)}) VALUES ({placeholders})", rows)

    # ---- reads --------------------------------------------------------
    def prediction_counts(self) -> dict[str, int]:
        row = self._read(
            "SELECT COUNT(*) AS total, COALESCE(SUM(is_suspicious), 0) AS suspicious, "
            "COALESCE(SUM(is_synthetic), 0) AS synthetic FROM predictions"
        )[0]
        return {"total": int(row["total"]), "suspicious": int(row["suspicious"]), "synthetic": int(row["synthetic"])}

    def request_stats(self) -> dict[str, Any]:
        row = self._read(
            "SELECT COUNT(*) AS total, COALESCE(SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END), 0) AS errors "
            "FROM requests"
        )[0]
        by_path = self._read(
            "SELECT path, COUNT(*) AS n FROM requests GROUP BY path ORDER BY n DESC LIMIT 20"
        )
        return {"total": int(row["total"]), "errors": int(row["errors"]), "by_path": {r["path"]: int(r["n"]) for r in by_path}}

    def prediction_latencies(self, limit: int = 5000) -> list[float]:
        rows = self._read(
            "SELECT latency_ms FROM requests WHERE path IN ('/predict', '/predict/batch') "
            "AND status_code < 400 ORDER BY id DESC LIMIT ?", (limit,)
        )
        return [float(r["latency_ms"]) for r in rows]

    def recent_predictions(self, limit: int = 200) -> list[dict[str, Any]]:
        rows = self._read("SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (int(limit),))
        out = []
        for r in rows:
            d = dict(r)
            d["is_synthetic"] = bool(d["is_synthetic"])
            d["is_suspicious"] = bool(d["is_suspicious"])
            out.append(d)
        return out
