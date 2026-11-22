"""Request and response contracts for the crop API."""

from pydantic import BaseModel, ConfigDict, Field

from src.input_config import INPUT_LIMITS


class CropInput(BaseModel):
    """Inclusive feature ranges computed from the raw dataset."""
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)

    N: float = Field(ge=INPUT_LIMITS["N"][0], le=INPUT_LIMITS["N"][1])
    P: float = Field(ge=INPUT_LIMITS["P"][0], le=INPUT_LIMITS["P"][1])
    K: float = Field(ge=INPUT_LIMITS["K"][0], le=INPUT_LIMITS["K"][1])
    temperature: float = Field(ge=INPUT_LIMITS["temperature"][0], le=INPUT_LIMITS["temperature"][1])
    humidity: float = Field(ge=INPUT_LIMITS["humidity"][0], le=INPUT_LIMITS["humidity"][1])
    ph: float = Field(ge=INPUT_LIMITS["ph"][0], le=INPUT_LIMITS["ph"][1])
    rainfall: float = Field(ge=INPUT_LIMITS["rainfall"][0], le=INPUT_LIMITS["rainfall"][1])


class CropProbability(BaseModel):
    crop: str = Field(min_length=1)
    probability: float = Field(ge=0, le=1, allow_inf_nan=False)


class CropResponse(BaseModel):
    recommended_crop: str = Field(min_length=1)
    top_3: list[CropProbability] = Field(min_length=1, max_length=3)


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool
