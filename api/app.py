"""FastAPI service for heart disease risk prediction."""
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field, field_validator

from src.predict import load_model, predict

DEFAULT_MODEL_PATH = "models/heart_model.joblib"
META_PATH = Path("models/metadata.json")

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("heart-api")

REQUESTS = Counter("api_requests_total", "HTTP requests", ["method", "path", "status"])
LATENCY = Histogram("api_request_latency_seconds", "Request latency", ["path"])
PREDICTIONS = Counter("api_predictions_total", "Predictions by class", ["label"])
ERRORS = Counter("api_prediction_errors_total", "Failed predictions")


def log_event(**fields):
    """One JSON object per line, easy to parse in Cloud Logging."""
    logger.info(json.dumps(fields, default=str))


class Patient(BaseModel):
    age: int = Field(..., ge=1, le=120, examples=[63])
    sex: int = Field(..., ge=0, le=1, examples=[1])
    cp: int = Field(..., ge=1, le=4, examples=[4])
    trestbps: float = Field(..., ge=50, le=300, examples=[145])
    chol: float = Field(..., ge=50, le=700, examples=[233])
    fbs: int = Field(..., ge=0, le=1, examples=[1])
    restecg: int = Field(..., ge=0, le=2, examples=[2])
    thalach: float = Field(..., ge=40, le=250, examples=[150])
    exang: int = Field(..., ge=0, le=1, examples=[0])
    oldpeak: float = Field(..., ge=0, le=10, examples=[2.3])
    slope: int = Field(..., ge=1, le=3, examples=[3])
    ca: int = Field(..., ge=0, le=3, examples=[0])
    thal: int = Field(..., ge=3, le=7, examples=[6])

    @field_validator("thal")
    @classmethod
    def thal_must_be_known(cls, v):
        if v not in (3, 6, 7):
            raise ValueError("thal must be 3, 6 or 7")
        return v


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
    app.state.model = load_model(Path(path))
    app.state.meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
    log_event(event="startup", model_path=path, model=app.state.meta.get("selected_run"))
    yield


app = FastAPI(title="Heart Disease Risk API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def observe(request: Request, call_next):
    start = time.perf_counter()
    request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["x-request-id"] = request_id
        return response
    finally:
        elapsed = time.perf_counter() - start
        route = request.scope.get("route")
        path = route.path if route else "unmatched"
        REQUESTS.labels(request.method, path, str(status)).inc()
        LATENCY.labels(path).observe(elapsed)
        log_event(event="request", request_id=request_id, method=request.method,
                  path=request.url.path, status=status, latency_ms=round(elapsed * 1000, 2))


@app.get("/")
def root():
    return {"service": "heart-disease-api", "docs": "/docs", "predict": "POST /predict"}


@app.get("/health")
def health(request: Request):
    if getattr(request.app.state, "model", None) is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ok", "model_loaded": True}


@app.post("/predict")
def predict_endpoint(patient: Patient, request: Request):
    record = patient.model_dump()
    try:
        result = predict(request.app.state.model, record)
    except Exception as exc:
        ERRORS.inc()
        log_event(event="prediction_error", error=str(exc))
        raise HTTPException(status_code=500, detail="prediction failed") from exc
    PREDICTIONS.labels(result["label"]).inc()
    log_event(event="prediction", input=record, **result)
    return {**result, "model_version": request.app.state.meta.get("selected_run", "unknown")}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
