from __future__ import annotations

from fastapi.testclient import TestClient


def test_demo_ui(app):
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "CareSignal" in response.text


def test_model_info(app):
    client = TestClient(app)
    response = client.get("/info")
    assert response.status_code == 200
    body = response.json()
    assert "threshold" in body
    assert len(body["features"]) == 7


def test_health(app):
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_version"]


def test_predict_valid_payload(app):
    client = TestClient(app)
    payload = {
        "age": 58,
        "sex": "male",
        "length_of_stay_days": 4.5,
        "prior_admissions": 1,
        "condition_count": 2,
        "procedure_count": 1,
        "days_since_last_discharge": 120.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["risk_score"] <= 1.0
    assert isinstance(body["predicted_readmission"], bool)


def test_predict_missing_field_returns_422(app):
    client = TestClient(app)
    response = client.post("/predict", json={"age": 40})
    assert response.status_code == 422
