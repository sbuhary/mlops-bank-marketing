from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

from src.data.load_data import prepare_data
from src.data.validate import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS
from src.features.preprocessing import build_preprocessor


MODEL_NAME = "bank-marketing-model"


def load_params(params_path: str | Path = "params.yaml") -> dict[str, Any]:
    with Path(params_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def configure_mlflow(params: dict[str, Any]) -> None:
    """Configure local MLflow by default, with optional DagsHub remote tracking."""
    tracking_uri = params["mlflow"].get("tracking_uri", "mlruns")
    username = os.getenv("DAGSHUB_USERNAME")
    token = os.getenv("DAGSHUB_TOKEN")

    if username and token:
        repo_owner = os.getenv("DAGSHUB_REPO_OWNER", username)
        repo_name = os.getenv("DAGSHUB_REPO_NAME", "mlops-bank-marketing")
        tracking_uri = f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"
        os.environ["MLFLOW_TRACKING_USERNAME"] = username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(params["mlflow"]["experiment_name"])


def build_models(params: dict[str, Any]) -> dict[str, Pipeline]:
    logistic_params = params["models"]["logistic_regression"]
    forest_params = params["models"]["random_forest"]

    return {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", LogisticRegression(**logistic_params)),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", RandomForestClassifier(**forest_params)),
            ]
        ),
    }


def split_features_target(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=[target]), df[target]


def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }


def load_training_data(params: dict[str, Any], params_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    processed_dir = Path(params["data"]["processed_dir"])
    train_path = processed_dir / "train.csv"
    test_path = processed_dir / "test.csv"
    if train_path.exists() and test_path.exists():
        return pd.read_csv(train_path), pd.read_csv(test_path)
    return prepare_data(params_path)


def train(params_path: str | Path = "params.yaml") -> dict[str, Any]:
    params = load_params(params_path)
    configure_mlflow(params)

    train_df, test_df = load_training_data(params, params_path)
    target = params["data"]["target"]
    X_train, y_train = split_features_target(train_df, target)
    X_test, y_test = split_features_target(test_df, target)

    model_dir = Path(params["artifacts"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path(params["artifacts"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    best_name = ""
    best_model: Pipeline | None = None
    best_metrics: dict[str, float] = {}
    all_metrics: dict[str, dict[str, float]] = {}
    selection_metric = params["training"]["selection_metric"]

    for model_name, model in build_models(params).items():
        with mlflow.start_run(run_name=model_name):
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            metrics = evaluate_predictions(y_test, predictions)
            all_metrics[model_name] = metrics

            # Tracking turns each experiment into an auditable model candidate.
            mlflow.log_params(params["models"][model_name])
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("numeric_features", ",".join(NUMERIC_COLUMNS))
            mlflow.log_param("categorical_features", ",".join(CATEGORICAL_COLUMNS))
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                model,
                name="model",
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            )

            # Promotion should be metric-driven, not whichever model ran last.
            if not best_metrics or metrics[selection_metric] > best_metrics[selection_metric]:
                best_name = model_name
                best_model = model
                best_metrics = metrics

    if best_model is None:
        raise RuntimeError("No model was trained.")

    model_path = model_dir / "best_model.pkl"
    joblib.dump(best_model, model_path)

    metadata = {
        "model_name": best_name,
        "registered_model_name": MODEL_NAME,
        "model_version": params["serving"]["model_version"],
        "selection_metric": selection_metric,
        "metrics": best_metrics,
        "all_metrics": all_metrics,
    }
    (model_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (reports_dir / "train_metrics.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(f"Best model: {best_name}")
    print(json.dumps(best_metrics, indent=2))
    return metadata


def main() -> None:
    train()


if __name__ == "__main__":
    main()
