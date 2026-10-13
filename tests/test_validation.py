import pytest

from tests.conftest import VALID_TXN

HEADERS = {"Content-Type": "application/json"}


def _without(field):
    return {k: v for k, v in VALID_TXN.items() if k != field}


@pytest.mark.parametrize("field", ["amount", "hour", "merchant_category", "distance_from_home_km",
                                   "txn_count_24h", "avg_amount_30d", "is_online", "is_foreign", "card_age_days"])
def test_missing_feature_rejected(client, field):
    resp = client.post("/predict", json=_without(field))
    assert resp.status_code == 422
    assert any(field in e["field"] for e in resp.json()["errors"])


@pytest.mark.parametrize("field,value", [
    ("amount", -5), ("amount", 0), ("amount", "abc"), ("amount", 10_000_000),
    ("hour", 24), ("hour", -1), ("hour", 3.5),
    ("merchant_category", 10), ("is_online", 2), ("txn_count_24h", -1),
    ("distance_from_home_km", -1), ("card_age_days", -3),
    ("transaction_id", "bad id!"), ("transaction_id", "x" * 100),
])
def test_invalid_values_rejected(client, field, value):
    resp = client.post("/predict", json={**VALID_TXN, field: value})
    assert resp.status_code == 422
    assert "Traceback" not in resp.text


def test_nan_and_infinity_rejected(client):
    for bad in ("NaN", "Infinity", "-Infinity"):
        body = '{"amount": %s, "hour": 1, "merchant_category": 1, "distance_from_home_km": 1,' \
               ' "txn_count_24h": 1, "avg_amount_30d": 1, "is_online": 0, "is_foreign": 0,' \
               ' "card_age_days": 10}' % bad
        assert client.post("/predict", content=body, headers=HEADERS).status_code == 422


def test_unknown_field_rejected(client):
    assert client.post("/predict", json={**VALID_TXN, "card_number": "4111111111111111"}).status_code == 422


def test_malformed_json_rejected(client):
    resp = client.post("/predict", content="{not valid json", headers=HEADERS)
    assert resp.status_code == 422
    assert "Traceback" not in resp.text


def test_empty_body_rejected(client):
    assert client.post("/predict", content="", headers=HEADERS).status_code == 422


def test_error_response_does_not_echo_input(client):
    resp = client.post("/predict", json={**VALID_TXN, "amount": "SECRET-VALUE"})
    assert "SECRET-VALUE" not in resp.text


def test_batch_validation(client):
    assert client.post("/predict/batch", json={"transactions": []}).status_code == 422
    assert client.post("/predict/batch", json={"transactions": [VALID_TXN, {"amount": 1}]}).status_code == 422
    too_many = {"transactions": [VALID_TXN] * 1001}
    assert client.post("/predict/batch", json=too_many).status_code == 422
    assert client.post("/predict/batch", json=[VALID_TXN]).status_code == 422


def test_demo_request_validation(client):
    assert client.post("/demo/generate", json={"n": 0}).status_code == 422
    assert client.post("/demo/generate", json={"n": 501}).status_code == 422
