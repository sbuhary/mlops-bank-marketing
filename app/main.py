from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.models.predict import DEFAULT_FEATURES, load_model, predict_one


MODEL_PATH = Path("models/best_model.pkl")
METADATA_PATH = Path("models/model_metadata.json")
LOG_PATH = Path("monitoring/prediction_logs.csv")

app = FastAPI(title="Bank Marketing Prediction API", version="1.0")
_model = None


class PredictionRequest(BaseModel):
    age: int = Field(..., ge=18, le=100)
    job: str = "admin"
    marital: str = "married"
    education: str = "secondary"
    balance: float = 0
    housing: Literal["yes", "no"] = "no"
    loan: Literal["yes", "no"] = "no"
    contact: str = "cellular"
    campaign: int = Field(1, ge=1)
    previous: int = Field(0, ge=0)
    poutcome: str = "unknown"


class PredictionResponse(BaseModel):
    prediction: int
    model_version: str


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="Model artifact not found. Run `python -m src.models.train` first.",
            )
        _model = load_model(MODEL_PATH)
    return _model


def get_model_version() -> str:
    if METADATA_PATH.exists():
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        return str(metadata.get("model_version", "1.0"))
    return "1.0"


def log_prediction(payload: dict, prediction: int, model_version: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    is_new_file = not LOG_PATH.exists()
    # Prediction logs are the first building block for drift and audit checks.
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_version": model_version,
        "prediction": prediction,
        **{key: payload.get(key, DEFAULT_FEATURES[key]) for key in DEFAULT_FEATURES},
    }

    with LOG_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if is_new_file:
            writer.writeheader()
        writer.writerow(row)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "MLOps Bank Marketing API running"}


@app.get("/health")
def health() -> dict[str, str]:
    model_status = "ready" if MODEL_PATH.exists() else "missing"
    return {"status": "ok", "model_status": model_status}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    payload = request.model_dump()
    model = get_model()
    prediction = predict_one(model, payload)
    model_version = get_model_version()
    log_prediction(payload, prediction, model_version)
    return PredictionResponse(prediction=prediction, model_version=model_version)
