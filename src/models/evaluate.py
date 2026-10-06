from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from src.models.train import evaluate_predictions, load_params, split_features_target


def evaluate(params_path: str | Path = "params.yaml") -> dict[str, float]:
    params = load_params(params_path)
    test_df = pd.read_csv(Path(params["data"]["processed_dir"]) / "test.csv")
    X_test, y_test = split_features_target(test_df, params["data"]["target"])

    model = joblib.load(Path(params["artifacts"]["model_dir"]) / "best_model.pkl")
    predictions = model.predict(X_test)
    metrics = evaluate_predictions(y_test, predictions)

    reports_dir = Path(params["artifacts"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    evaluate()


if __name__ == "__main__":
    main()
