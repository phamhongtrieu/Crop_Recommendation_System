"""Load a trusted team artifact once and pass raw features to its predict method."""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from backend.schemas import CropInput, CropProbability, CropResponse

logger = logging.getLogger(__name__)
FEATURE_NAMES = ("N", "P", "K", "temperature", "humidity", "ph", "rainfall")
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "crop_model.joblib"


class ModelUnavailableError(Exception):
    """The artifact is absent or could not be loaded."""


class ModelService:
    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = model_path
        self._model = None
        self._load_attempted = False

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Attempt once per service instance, including missing or invalid artifacts."""
        if self._load_attempted:
            return
        self._load_attempted = True
        try:
            model = joblib.load(self.model_path)
            if not callable(getattr(model, "predict", None)):
                raise ValueError("Artifact must expose predict(X)")
            if not callable(getattr(model, "predict_proba", None)):
                raise ValueError("Artifact must expose predict_proba(X)")
            names = getattr(model, "feature_names_in_", None)
            if names is not None and list(names) != list(FEATURE_NAMES):
                raise ValueError("Artifact feature names/order do not match the API contract")
            count = getattr(model, "n_features_in_", None)
            if count is not None and count != len(FEATURE_NAMES):
                raise ValueError("Artifact must accept seven features")
            self._model = model
            logger.info("Crop model loaded from %s", self.model_path)
        except FileNotFoundError:
            logger.warning("Crop model is missing at %s; prediction remains unavailable", self.model_path)
        except Exception:
            logger.exception("Unable to load crop model from %s", self.model_path)

    def predict(self, crop: CropInput) -> CropResponse:
        if self._model is None:
            raise ModelUnavailableError("Crop model is not ready")
        values = crop.model_dump()
        frame = pd.DataFrame([[values[name] for name in FEATURE_NAMES]], columns=FEATURE_NAMES)
        predictions = self._model.predict(frame)
        # A single-row classifier must return one nonempty crop label, not an encoded ID.
        if isinstance(predictions, (str, bytes)) or len(predictions) != 1:
            raise ValueError("Expected one crop prediction")
        label = predictions[0]
        if not isinstance(label, str) or not label.strip():
            raise ValueError("Expected a nonempty crop name; encoded labels need a decoder")
        classes = np.asarray(self._model.classes_, dtype=object)
        probabilities = np.asarray(self._model.predict_proba(frame))
        if classes.ndim != 1 or len(classes) == 0:
            raise ValueError("Expected a nonempty one-dimensional class list")
        if any(not isinstance(name, str) or not name.strip() for name in classes):
            raise ValueError("Expected crop names in classes_")
        if len(set(classes)) != len(classes) or label not in classes:
            raise ValueError("Class labels must be unique and include the prediction")
        if probabilities.shape != (1, len(classes)) or probabilities.dtype.kind not in "fiu":
            raise ValueError("Expected one numeric probability per class for one sample")
        if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
            raise ValueError("Probabilities must be finite and between zero and one")
        # Stable sorting preserves classes_ order for ties; predict() remains authoritative.
        ranking = sorted(range(len(classes)), key=lambda index: -float(probabilities[0, index]))[:3]
        return CropResponse(
            recommended_crop=label,
            top_3=[CropProbability(crop=classes[index], probability=float(probabilities[0, index]))
                   for index in ranking],
        )
