"""FastAPI entry point: python -m uvicorn backend.app:app --port 8000."""

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.model_service import ModelService, ModelUnavailableError
from backend.schemas import CropInput, CropResponse, HealthResponse

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        service = ModelService()
        service.load()
        application.state.model_service = service
        yield

    application = FastAPI(
        title="Crop Recommendation API", version="1.0.0", lifespan=lifespan
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @application.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        # Omit raw input/context: NaN and Infinity cannot be serialized as JSON.
        errors = [
            {"loc": error["loc"], "msg": error["msg"], "type": error["type"]}
            for error in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": errors})

    @application.get("/")
    def root() -> dict[str, str]:
        return {"name": "Crop Recommendation API", "version": "1.0.0", "docs": "/docs"}

    @application.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        return HealthResponse(model_loaded=request.app.state.model_service.model_loaded)

    @application.post(
        "/predict",
        response_model=CropResponse,
        responses={503: {"description": "Model is not ready"},
                   500: {"description": "Internal prediction error"}},
    )
    def predict(crop: CropInput, request: Request) -> CropResponse:
        try:
            return request.app.state.model_service.predict(crop)
        except ModelUnavailableError:
            raise HTTPException(
                status_code=503,
                detail="Crop model is not ready. Please contact the backend team and retry after model deployment.",
            ) from None
        except Exception:
            logger.exception("Crop prediction failed")
            raise HTTPException(status_code=500, detail="Internal prediction error.") from None

    return application


app = create_app()
