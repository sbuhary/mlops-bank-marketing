from __future__ import annotations

from pathlib import Path

import mlflow

from src.models.train import MODEL_NAME, configure_mlflow, load_params


def register_latest_model(params_path: str | Path = "params.yaml") -> str:
    """Register the latest logged MLflow model for manual stage promotion."""
    params = load_params(params_path)
    configure_mlflow(params)
    experiment = mlflow.get_experiment_by_name(params["mlflow"]["experiment_name"])
    if experiment is None:
        raise RuntimeError("Train a model before registering it.")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.f1 DESC"],
        max_results=1,
    )
    if runs.empty:
        raise RuntimeError("No MLflow runs found for registration.")

    run_id = runs.iloc[0]["run_id"]
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri, MODEL_NAME)
    print(f"Registered {MODEL_NAME} version {result.version}")
    return str(result.version)


if __name__ == "__main__":
    register_latest_model()
