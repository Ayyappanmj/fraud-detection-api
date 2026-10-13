from app.model_service import ModelService


def test_root_returns_info(client):
    body = client.get("/").json()
    assert "Fraud Detection" in body["message"]
    assert body["docs"] == "/docs"
    assert "SYNTHETIC" in body["disclaimer"]


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"
    assert resp.json()["model_loaded"] is True


def test_health_unhealthy_without_model(client_no_model):
    resp = client_no_model.get("/health")
    assert resp.status_code == 503
    assert resp.json()["status"] == "unhealthy"


def test_predict_returns_503_without_model(client_no_model):
    from tests.conftest import VALID_TXN
    resp = client_no_model.post("/predict", json=VALID_TXN)
    assert resp.status_code == 503
    assert "Traceback" not in resp.text


def test_model_info(client):
    body = client.get("/model-info").json()
    assert body["model_type"] == "IsolationForest"
    assert "amount" in body["feature_schema"]
    assert body["model_version"].startswith("1.0.0-")
    assert "NOT a calibrated probability" in body["score_definition"]
    assert "isolation_forest" in body["evaluation"]


def test_docs_available(client):
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200


def test_request_id_generated_and_echoed(client):
    assert client.get("/").headers["X-Request-ID"]
    assert client.get("/", headers={"X-Request-ID": "abc-123"}).headers["X-Request-ID"] == "abc-123"
    # unsafe ids are replaced, not reflected
    unsafe = client.get("/", headers={"X-Request-ID": "bad id with spaces"}).headers["X-Request-ID"]
    assert unsafe != "bad id with spaces"


def test_missing_artifacts_do_not_crash(tmp_path):
    service = ModelService(tmp_path / "nothing-here")
    assert service.load() is False
    assert service.is_loaded is False
    assert service.load_error == "model artifacts not found"


def test_tampered_model_is_refused(model_dir, tmp_path):
    import shutil
    copy = tmp_path / "copy"
    shutil.copytree(model_dir, copy)
    with open(copy / "model.joblib", "ab") as handle:
        handle.write(b"tamper")
    service = ModelService(copy)
    assert service.load() is False
    assert service.load_error == "model checksum mismatch"
