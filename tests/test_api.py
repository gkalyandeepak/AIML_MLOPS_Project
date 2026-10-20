import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from src.features import build_pipeline
from src.predict import SAMPLE


@pytest.fixture
def client(tmp_path, monkeypatch, X, y):
    model = build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    monkeypatch.setenv("MODEL_PATH", str(path))
    from api.app import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_predict_valid(client):
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1.0
    assert 0.0 <= body["probability_disease"] <= 1.0
    assert "model_version" in body


def test_predict_out_of_range_rejected(client):
    r = client.post("/predict", json={**SAMPLE, "age": 500})
    assert r.status_code == 422


def test_predict_invalid_thal_rejected(client):
    r = client.post("/predict", json={**SAMPLE, "thal": 5})
    assert r.status_code == 422


def test_predict_missing_field_rejected(client):
    bad = {k: v for k, v in SAMPLE.items() if k != "chol"}
    assert client.post("/predict", json=bad).status_code == 422


def test_metrics_exposed(client):
    client.post("/predict", json=SAMPLE)
    text = client.get("/metrics").text
    assert "api_requests_total" in text
    assert "api_predictions_total" in text
