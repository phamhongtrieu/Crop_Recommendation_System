"""Request and response contracts for the crop API."""

from pydantic import BaseModel, ConfigDict, Field


class CropInput(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)

    N: float = Field(ge=0)
    P: float = Field(ge=0)
    K: float = Field(ge=0)
    temperature: float
    humidity: float = Field(ge=0, le=100)
    ph: float = Field(ge=0, le=14)
    rainfall: float = Field(ge=0)


class CropResponse(BaseModel):
    recommended_crop: str = Field(min_length=1)


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool
