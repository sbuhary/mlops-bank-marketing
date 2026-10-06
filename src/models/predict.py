from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd


DEFAULT_FEATURES = {
    "age": 40,
    "job": "admin",
    "marital": "married",
    "education": "secondary",
    "balance": 5000,
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "campaign": 2,
    "previous": 1,
    "poutcome": "unknown",
}


def load_model(model_path: str | Path = "models/best_model.pkl") -> Any:
    return joblib.load(model_path)


def normalize_features(payload: dict[str, Any]) -> pd.DataFrame:
    features = {**DEFAULT_FEATURES, **payload}
    return pd.DataFrame([features], columns=list(DEFAULT_FEATURES.keys()))


def predict_one(model: Any, payload: dict[str, Any]) -> int:
    frame = normalize_features(payload)
    return int(model.predict(frame)[0])
