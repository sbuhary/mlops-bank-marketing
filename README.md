# MLOps Bank Marketing

End-to-end MLOps learning project for predicting whether a banking customer will subscribe to a marketing offer.

## Architecture

```text
data/raw -> validation -> data/processed -> training -> MLflow/DagsHub
        -> model registry -> FastAPI serving -> Docker/CI -> Azure deployment
        -> monitoring logs -> drift analysis foundation
```

## Project Overview

This repository demonstrates a complete machine learning lifecycle with Python 3.11, scikit-learn, MLflow, DVC, FastAPI, Docker, GitHub Actions, and Azure deployment preparation.

The dataset follows the UCI Bank Marketing structure and includes numeric and categorical customer/contact features. The training code compares Logistic Regression and Random Forest models, selects the best model by F1 score, and saves it as `models/best_model.pkl`.

## Local Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux, activate with `source .venv/bin/activate`.

## Run The Complete Pipeline

From a fresh checkout:

```bash
pip install -r requirements.txt
python -m src.data.load_data
python -m src.models.train
python -m src.models.evaluate
```

With DVC:

```bash
dvc repro
```

The pipeline creates processed splits, trains models, logs MLflow runs, writes metrics under `reports/`, and saves the best model under `models/`.

## MLflow Usage

Local runs use the SQLite tracking backend configured in `params.yaml`.

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

For DagsHub tracking, set environment variables before training:

```bash
set DAGSHUB_USERNAME=<your-user>
set DAGSHUB_TOKEN=<your-token>
```

PowerShell users can use `$env:DAGSHUB_USERNAME="..."`.

## Model Registry

The registry model name is `bank-marketing-model`.

```bash
python -m src.models.register
```

New versions start in `None`, can be promoted to `Staging` after validation, and then to `Production` after approval. See `configs/model_registry.md`.

## API Usage

Start the service after training:

```bash
uvicorn app.main:app --reload
```

Health check:

```bash
curl http://localhost:8000/health
```

Prediction:

```bash
curl -X POST http://localhost:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"age\":40,\"job\":\"admin\",\"balance\":5000,\"housing\":\"yes\",\"loan\":\"no\"}"
```

Response:

```json
{
  "prediction": 1,
  "model_version": "1.0"
}
```

## Docker Usage

Train the model locally first so `models/best_model.pkl` exists, then run:

```bash
docker compose up --build
```

The API is exposed at `http://localhost:8000`.

## CI/CD

GitHub Actions workflows live in `.github/workflows/`:

- `ci.yml`: install dependencies, run tests, train model
- `docker-build.yml`: build and inspect the Docker image

## Azure Deployment Path

The `azure/` folder includes guidance and sample YAML for:

- Azure Container Registry for storing Docker images
- Azure Machine Learning for managed training and model registration
- Azure Kubernetes Service for scalable API deployment

## Monitoring Foundation

Every API prediction is appended to `monitoring/prediction_logs.csv` with timestamp, model version, prediction, and input features. This file is the starting point for data drift, model drift, and operational monitoring.

## MLOps Lifecycle

This project intentionally separates data validation, preprocessing, training, evaluation, serving, deployment, and monitoring. That separation makes experiments reproducible, model promotion auditable, and production behavior easier to test.
