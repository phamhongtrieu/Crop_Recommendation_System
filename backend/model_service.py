"""Load a trusted team artifact once and pass raw features to its predict method."""

import logging
from pathlib import Path

import joblib
import pandas as pd

from backend.schemas import CropInput

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
            names = getattr(model, "feature_names_in_", None)
            if names is not None and list(names) != list(FEATURE_NAMES):
                raise ValueError("Artifact feature names/order do not match the API contract")
            count = getattr(model, "n_features_in_", None)
            if count is not None and count != len(FEATURE_NAMES):
                raise ValueError("Artifact must accept seven features")
            self._model = model
        except FileNotFoundError:
            logger.warning("Crop model is missing; prediction remains unavailable")
        except Exception:
            logger.exception("Unable to load crop model")

    def predict(self, crop: CropInput) -> str:
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
        return label
