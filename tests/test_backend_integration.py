"""Read-only integration checks against the trusted model supplied by the ML team."""

import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

from backend.app import create_app
from backend.model_service import FEATURE_NAMES, MODEL_PATH

SAMPLE = {
    "N": 90, "P": 42, "K": 43, "temperature": 20.87974371,
    "humidity": 82.00274423, "ph": 6.502985292, "rainfall": 202.9355362,
}


@pytest.fixture(scope="module")
def real_client():
    if not MODEL_PATH.is_file():
        pytest.skip("Real-model integration requires models/crop_model.joblib")
    with TestClient(create_app()) as client:
        yield client


def test_real_model_contract(real_client):
    metadata = json.loads(MODEL_PATH.with_name("model_metadata.json").read_text(encoding="utf-8"))
    service = real_client.app.state.model_service
    assert service.model_loaded
    pipeline = service._model
    assert isinstance(pipeline, Pipeline)
    assert isinstance(pipeline.named_steps["model"], GaussianNB)
    assert pipeline.named_steps["scaler"] == "passthrough"
    assert metadata["model_file"] == MODEL_PATH.name
    assert metadata["estimator"] == "GaussianNB"
    assert metadata["features"] == list(FEATURE_NAMES)
    assert list(pipeline.named_steps["model"].feature_names_in_) == list(FEATURE_NAMES)
    assert pipeline.classes_.tolist() == metadata["classes"]


def test_real_model_health(real_client):
    response = real_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_real_model_predicts_rice_from_raw_values(real_client):
    # The API must order columns itself; no separate scaler is applied.
    response = real_client.post("/predict", json=dict(reversed(list(SAMPLE.items()))))
    assert response.status_code == 200
    result = response.json()
    assert result["recommended_crop"] == "rice"
    assert len(result["top_3"]) == 3
    assert result["top_3"][0]["crop"] == result["recommended_crop"]
    model = real_client.app.state.model_service._model
    frame = pd.DataFrame([SAMPLE], columns=FEATURE_NAMES)
    probabilities = model.predict_proba(frame)[0]
    expected = sorted(zip(model.classes_, probabilities), key=lambda pair: -pair[1])[:3]
    assert result["top_3"] == [
        {"crop": crop, "probability": float(probability)} for crop, probability in expected
    ]
    assert all(0 <= item["probability"] <= 1 for item in result["top_3"])


def test_real_model_still_validates_input(real_client):
    response = real_client.post("/predict", json={**SAMPLE, "humidity": 101})
    assert response.status_code == 422
