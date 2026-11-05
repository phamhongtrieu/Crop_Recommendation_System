"""Backend contract tests; never read or write a real model artifact."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from backend.app import create_app
from backend.model_service import FEATURE_NAMES, MODEL_PATH, ModelService

SAMPLE = dict(N=90, P=42, K=43, temperature=20.87, humidity=82.0, ph=6.5, rainfall=202.9)


@pytest.fixture
def loader(monkeypatch):
    mock = Mock(side_effect=FileNotFoundError)
    monkeypatch.setattr("backend.model_service.joblib.load", mock)
    return mock


@pytest.fixture
def client(loader):
    with TestClient(create_app()) as test_client:
        yield test_client


def model_returning(labels):
    model = Mock(spec=["predict"])
    model.predict.return_value = labels
    return model


def test_root_health_and_missing_model(client, loader):
    assert client.get("/").json()["name"] == "Crop Recommendation API"
    assert client.get("/health").json() == {"status": "ok", "model_loaded": False}
    response = client.post("/predict", json=SAMPLE)
    assert response.status_code == 503
    assert "not ready" in response.json()["detail"]
    client.get("/health")
    loader.assert_called_once_with(MODEL_PATH)


@pytest.mark.parametrize("field,value", [
    ("N", -1), ("P", -1), ("K", -1), ("rainfall", -0.1),
    ("humidity", -0.1), ("humidity", 100.1), ("ph", -0.1), ("ph", 14.1),
    ("temperature", "warm"), ("N", True), ("N", "90"), ("ph", None),
])
def test_invalid_values(client, field, value):
    assert client.post("/predict", json={**SAMPLE, field: value}).status_code == 422


@pytest.mark.parametrize("field", FEATURE_NAMES)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_values_return_serializable_422(client, field, value):
    response = client.post(
        "/predict", content=json.dumps({**SAMPLE, field: value}),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", field]


@pytest.mark.parametrize("payload", [{}, [], None, {**SAMPLE, "extra": 1}])
def test_invalid_payload(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_malformed_json(client):
    assert client.post("/predict", content="{", headers={"Content-Type": "application/json"}).status_code == 422


@pytest.mark.parametrize("humidity,ph", [(0, 0), (100, 14)])
def test_inclusive_bounds(client, humidity, ph):
    payload = {**SAMPLE, "N": 0, "P": 0, "K": 0, "rainfall": 0,
               "temperature": -10, "humidity": humidity, "ph": ph}
    assert client.post("/predict", json=payload).status_code == 503


def test_prediction_feature_order_and_single_load(loader):
    model = model_returning(np.array(["rice"]))
    loader.side_effect = None
    loader.return_value = model
    with TestClient(create_app()) as client:
        assert client.get("/health").json()["model_loaded"] is True
        # Request insertion order must not affect the model input.
        for _ in range(2):
            response = client.post("/predict", json=dict(reversed(list(SAMPLE.items()))))
            assert response.status_code == 200
            assert response.json() == {"recommended_crop": "rice"}
        frame = model.predict.call_args.args[0]
        assert list(frame.columns) == list(FEATURE_NAMES)
        assert frame.shape == (1, 7)
        assert frame.iloc[0].tolist() == [SAMPLE[name] for name in FEATURE_NAMES]
    loader.assert_called_once()


def test_pipeline_handles_its_own_preprocessing(loader):
    class StubClassifier(BaseEstimator):
        def fit(self, X, y=None):
            return self

        def predict(self, X):
            self.received_frame = X
            return ["maize"]

    estimator = StubClassifier()
    estimator.fitted_ = True
    transform = FunctionTransformer(lambda frame: frame * 2)
    loader.side_effect = None
    loader.return_value = Pipeline([("transform", transform), ("classifier", estimator)])
    with TestClient(create_app()) as client:
        assert client.post("/predict", json=SAMPLE).json() == {"recommended_crop": "maize"}
    assert estimator.received_frame.iloc[0].tolist() == [2 * SAMPLE[n] for n in FEATURE_NAMES]


def test_prediction_error_does_not_expose_details(loader):
    model = model_returning(["rice"])
    model.predict.side_effect = RuntimeError("secret artifact path and traceback")
    loader.side_effect = None
    loader.return_value = model
    with TestClient(create_app()) as client:
        response = client.post("/predict", json=SAMPLE)
        assert response.status_code == 500
        assert response.json() == {"detail": "Internal prediction error."}


@pytest.mark.parametrize("labels", [[], ["rice", "maize"], [0], [None], [""], [" "], "rice", [["rice"]]])
def test_invalid_model_output(loader, labels):
    loader.side_effect = None
    loader.return_value = model_returning(labels)
    with TestClient(create_app()) as client:
        assert client.post("/predict", json=SAMPLE).status_code == 500


@pytest.mark.parametrize("artifact", [object(), "corrupt", "wrong_columns", "wrong_count"])
def test_bad_artifact_does_not_prevent_startup(loader, artifact):
    loader.side_effect = None
    if artifact == "corrupt":
        loader.side_effect = ValueError("private loader details")
    elif artifact == "wrong_columns":
        artifact = model_returning(["rice"])
        artifact.feature_names_in_ = list(reversed(FEATURE_NAMES))
    elif artifact == "wrong_count":
        artifact = model_returning(["rice"])
        artifact.n_features_in_ = 6
    loader.return_value = artifact
    with TestClient(create_app()) as client:
        assert client.get("/health").json()["model_loaded"] is False
        assert client.post("/predict", json=SAMPLE).status_code == 503


def test_model_path_independent_of_working_directory(loader, monkeypatch):
    original_directory = Path.cwd()
    expected_path = Path(__file__).resolve().parents[1] / "models" / "crop_model.pkl"
    # Avoid pytest's shared %TEMP%/pytest-of-Admin, which may belong to a
    # different Windows account when IDE and sandbox use the same TEMP path.
    with TemporaryDirectory(prefix="model-path-", dir=Path(__file__).resolve().parent) as directory:
        temporary_directory = Path(directory)
        # Restore cwd before TemporaryDirectory removes it (required on Windows).
        with monkeypatch.context() as context:
            context.chdir(temporary_directory)
            assert Path.cwd() == temporary_directory
            assert Path.cwd() != original_directory
            service = ModelService()
            service.load()
            service.load()
            assert service.model_path.is_absolute()
            assert service.model_path == expected_path
            loader.assert_called_once_with(expected_path)
        assert Path.cwd() == original_directory
    assert not temporary_directory.exists()


@pytest.mark.parametrize("origin", ["http://localhost:8501", "http://127.0.0.1:8501"])
def test_allowed_cors(client, origin):
    response = client.options("/predict", headers={
        "Origin": origin, "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type",
    })
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin


def test_other_origin_rejected(client):
    response = client.options("/predict", headers={
        "Origin": "https://other.example", "Access-Control-Request-Method": "POST",
    })
    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers


def test_docs_and_openapi(client):
    assert client.get("/docs").status_code == 200
    schema = client.get("/openapi.json").json()
    assert set(schema["paths"]) == {"/", "/health", "/predict"}
    assert schema["components"]["schemas"]["CropInput"]["required"] == list(FEATURE_NAMES)
