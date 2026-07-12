import pandas as pd
import pytest
from fastapi.testclient import TestClient

from stroke_risk.app import main as app_main

VALID_PATIENT = {
    "id": 1,
    "age": 67,
    "hypertension": 0,
    "heart_disease": 1,
    "avg_glucose_level": 228.69,
    "bmi": 36.6,
    "gender": "male",
    "ever_married": "yes",
    "work_type": "private",
    "residence_type": "urban",
    "smoking_status": "formerly_smoked",
}


class StubModel:
    def predict(self, X: pd.DataFrame):
        return [1]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app_main, "load_model", lambda: StubModel())
    with TestClient(app_main.app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_prediction_for_valid_patient(client):
    response = client.post("/predict", json=VALID_PATIENT)

    assert response.status_code == 200
    assert response.json() == {"patient_id": 1, "prediction": "High Risk of Stroke"}


def test_predict_rejects_invalid_patient(client):
    response = client.post("/predict", json={**VALID_PATIENT, "gender": "invalid"})

    assert response.status_code == 422


def test_predict_returns_503_when_no_model_available(monkeypatch):
    def raise_not_found():
        raise Exception("Registered Model with name=stroke-risk not found")

    monkeypatch.setattr(app_main, "load_model", raise_not_found)

    with TestClient(app_main.app) as client:
        health_response = client.get("/health")
        predict_response = client.post("/predict", json=VALID_PATIENT)

    assert health_response.status_code == 200
    assert predict_response.status_code == 503


def test_reload_picks_up_a_newly_promoted_model(monkeypatch):
    def raise_not_found():
        raise Exception("Registered Model with name=stroke-risk not found")

    monkeypatch.setattr(app_main, "load_model", raise_not_found)

    with TestClient(app_main.app) as client:
        assert client.post("/predict", json=VALID_PATIENT).status_code == 503

        monkeypatch.setattr(app_main, "load_model", lambda: StubModel())
        reload_response = client.post("/reload")
        predict_response = client.post("/predict", json=VALID_PATIENT)

    assert reload_response.status_code == 200
    assert reload_response.json() == {"status": "reloaded"}
    assert predict_response.status_code == 200


def test_reload_returns_503_when_no_model_available(client, monkeypatch):
    def raise_not_found():
        raise Exception("Registered Model with name=stroke-risk not found")

    monkeypatch.setattr(app_main, "load_model", raise_not_found)

    response = client.post("/reload")

    assert response.status_code == 503
