from tests.conftest import SUSPICIOUS_TXN, VALID_TXN


def test_predict_single(client):
    resp = client.post("/predict", json=VALID_TXN)
    assert resp.status_code == 200
    body = resp.json()
    assert body["transaction_id"] == "test-001"
    assert isinstance(body["anomaly_score"], float)
    assert body["classification"] in {"suspicious", "normal"}
    assert body["is_suspicious"] == (body["classification"] == "suspicious")
    assert body["threshold"] > 0
    assert body["model_version"].startswith("1.0.0-")
    assert body["processing_time_ms"] >= 0
    assert body["request_id"]


def test_predict_is_deterministic(client):
    first = client.post("/predict", json=VALID_TXN).json()["anomaly_score"]
    second = client.post("/predict", json=VALID_TXN).json()["anomaly_score"]
    assert first == second


def test_obvious_anomaly_scores_higher_and_is_flagged(client):
    normal = client.post("/predict", json=VALID_TXN).json()
    odd = client.post("/predict", json=SUSPICIOUS_TXN).json()
    assert odd["anomaly_score"] > normal["anomaly_score"]
    assert odd["classification"] == "suspicious"


def test_transaction_id_generated_when_missing(client):
    payload = {k: v for k, v in VALID_TXN.items() if k != "transaction_id"}
    assert client.post("/predict", json=payload).json()["transaction_id"].startswith("txn-")


def test_batch_predict(client):
    payload = {"transactions": [VALID_TXN, SUSPICIOUS_TXN, {**VALID_TXN, "transaction_id": "test-003"}]}
    resp = client.post("/predict/batch", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 3
    assert [p["transaction_id"] for p in body["predictions"]] == ["test-001", "test-sus", "test-003"]
    assert body["suspicious_count"] == sum(p["is_suspicious"] for p in body["predictions"])
    assert body["suspicious_count"] >= 1


def test_metrics_reflect_traffic(client):
    client.post("/predict", json=VALID_TXN)
    client.post("/predict", json=SUSPICIOUS_TXN)
    client.post("/predict", json={"amount": -1})  # invalid -> counted as error
    body = client.get("/metrics").json()
    assert body["predictions"]["total"] == 2
    assert body["predictions"]["suspicious"] >= 1
    assert body["requests"]["total"] == 3
    assert body["requests"]["errors"] == 1
    assert body["latency_ms"]["count"] == 2
    assert body["latency_ms"]["p95_ms"] is None  # too few observations


def test_recent_predictions(client):
    client.post("/predict", json=VALID_TXN)
    rows = client.get("/predictions/recent?limit=5").json()["predictions"]
    assert len(rows) == 1
    assert rows[0]["transaction_id"] == "test-001"
    assert rows[0]["is_synthetic"] is False


def test_drift_insufficient_then_warning(client):
    assert client.get("/drift").json()["status"] == "insufficient_data"
    assert client.post("/demo/generate", json={"n": 300, "seed": 11}).status_code == 200
    assert client.get("/drift").json()["status"] == "no_drift"
    assert client.post("/demo/generate", json={"n": 500, "seed": 12, "drift": True}).status_code == 200
    drift = client.get("/drift").json()
    assert drift["status"] == "drift_warning"
    assert "amount" in drift["drifted_features"]


def test_demo_generate_marks_synthetic(client):
    body = client.post("/demo/generate", json={"n": 20, "seed": 1}).json()
    assert body["generated"] == 20 and body["synthetic"] is True
    rows = client.get("/predictions/recent?limit=20").json()["predictions"]
    assert all(r["is_synthetic"] for r in rows)


def test_unexpected_error_is_hidden(app):
    from fastapi.testclient import TestClient

    def boom(_records):
        raise RuntimeError("super-secret-internal-detail")

    app.state.model_service.predict = boom
    resp = TestClient(app, raise_server_exceptions=False).post("/predict", json=VALID_TXN)
    assert resp.status_code == 500
    assert resp.json()["detail"] == "Internal server error"
    assert "super-secret" not in resp.text and "Traceback" not in resp.text
