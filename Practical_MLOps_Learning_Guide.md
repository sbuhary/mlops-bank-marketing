# Practical MLOps Learning Guide

## From Model Training to Production Deployment and Monitoring

This guide is a hands-on MLOps learning path based on the workshop
setup. The original workshop uses three main cloud tools:

-   GitHub: source control, Codespaces, GitHub Actions
-   DagsHub: MLflow experiment tracking, model registry and data
    versioning
-   Azure: cloud platform and deployment environment

The workshop setup requires GitHub, Azure for Students and DagsHub
accounts. It uses cloud-based tools and requires no local software
installation.

------------------------------------------------------------------------

# 1. MLOps Overview

Traditional ML workflow:

    Data
     |
     v
    Exploration
     |
     v
    Training
     |
     v
    Model File
     |
     v
    Manual Deployment
     |
     v
    Prediction

MLOps workflow:

                 GitHub
                    |
                    v
           Code + Version Control
                    |
                    v
            Training Pipeline
                    |
          +---------+----------+
          |                    |
          v                    v
       MLflow              DVC/Data
     Tracking             Versioning
          |
          v
     Model Registry
          |
          v
     Azure Deployment
          |
          v
     Monitoring + Retraining

MLOps connects software engineering practices with machine learning
lifecycle management.

------------------------------------------------------------------------

# 2. Create Required Accounts

## 2.1 GitHub Account

Create:

https://github.com

Use your university email.

Apply for:

https://education.github.com/pack

Benefits:

-   GitHub Codespaces hours
-   GitHub Actions minutes

------------------------------------------------------------------------

## 2.2 Azure for Students

Create:

https://azure.microsoft.com/en-us/free/students

Requirements:

-   University email
-   Student verification

Expected:

-   Azure for Students subscription
-   USD 100 credits

Verify:

``` bash
az account show --query name -o tsv
```

Expected:

    Azure for Students

------------------------------------------------------------------------

## 2.3 DagsHub

Create:

https://dagshub.com

Login using GitHub.

DagsHub provides:

-   MLflow tracking server
-   Dataset storage
-   Model registry

------------------------------------------------------------------------

# 3. Create Project Repository

Example project:

    mlops-bank-marketing

Recommended structure:

    mlops-bank-marketing/
    |
    ├── data/
    │   ├── raw/
    │   └── processed/
    |
    ├── notebooks/
    |
    ├── src/
    │   ├── data_loader.py
    │   ├── train.py
    │   ├── evaluate.py
    │   └── predict.py
    |
    ├── models/
    |
    ├── tests/
    |
    ├── requirements.txt
    ├── Dockerfile
    ├── Makefile
    ├── .github/
    │   └── workflows/
    │       └── ci.yml
    └── README.md

------------------------------------------------------------------------

# 4. Git Workflow

Clone repository:

``` bash
git clone https://github.com/<username>/mlops-bank-marketing.git

cd mlops-bank-marketing
```

Create branch:

``` bash
git checkout -b feature/training-pipeline
```

Commit:

``` bash
git add .
git commit -m "Add training pipeline"
```

Push:

``` bash
git push origin feature/training-pipeline
```

------------------------------------------------------------------------

# 5. GitHub Codespaces

Open:

    Repository
     -> Code
     -> Codespaces
     -> Create codespace

The environment runs in browser-based VS Code.

Test:

``` bash
python --version
```

------------------------------------------------------------------------

# 6. Python Environment

requirements.txt

    pandas
    numpy
    scikit-learn
    mlflow
    dagshub
    dvc
    fastapi
    uvicorn
    pytest

Install:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

# 7. Data Version Control with DVC

Initialize:

``` bash
dvc init
```

Add dataset:

``` bash
dvc add data/raw/bank.csv
```

Commit:

``` bash
git add .
git commit -m "Add dataset tracking"
```

DVC creates:

    bank.csv.dvc

The actual data can be stored separately.

------------------------------------------------------------------------

# 8. Model Training Example

src/train.py

``` python
import pandas as pd
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


data = pd.read_csv(
    "data/raw/bank.csv"
)

X = data.drop("target", axis=1)
y = data["target"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = LogisticRegression(
    max_iter=1000
)


with mlflow.start_run():

    model.fit(
        X_train,
        y_train
    )

    prediction = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        prediction
    )

    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    mlflow.sklearn.log_model(
        model,
        "model"
    )

print("Training completed")
```

Run:

``` bash
python src/train.py
```

------------------------------------------------------------------------

# 9. DagsHub MLflow Tracking

Configure:

``` bash
export MLFLOW_TRACKING_URI=https://dagshub.com/<username>/mlops-bank-marketing.mlflow
```

Authenticate:

``` bash
export DAGSHUB_TOKEN=<your_token>
```

Example:

``` python
import dagshub

dagshub.init(
    repo_owner="<username>",
    repo_name="mlops-bank-marketing",
    mlflow=True
)
```

Now every experiment is tracked.

Tracked information:

    Experiment
     |
     +-- Parameters
     |
     +-- Metrics
     |
     +-- Artifacts
     |
     +-- Model

------------------------------------------------------------------------

# 10. Model Registry

A trained model moves through stages:

    Training
       |
       v
    Experiment
       |
       v
    Registered Model
       |
       v
    Staging
       |
       v
    Production

Example:

``` python
mlflow.register_model(
    "runs:/RUN_ID/model",
    "bank-marketing-model"
)
```

------------------------------------------------------------------------

# 11. Create Prediction API

Install:

``` bash
pip install fastapi uvicorn
```

app.py

``` python
from fastapi import FastAPI
import mlflow.pyfunc


app = FastAPI()

model = mlflow.pyfunc.load_model(
    "models:/bank-marketing-model/Production"
)


@app.post("/predict")
def predict(data: dict):

    result = model.predict(
        [data]
    )

    return {
        "prediction": result[0]
    }
```

Run:

``` bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Test:

    POST /predict

------------------------------------------------------------------------

# 12. Docker Deployment

Dockerfile:

``` dockerfile
FROM python:3.11

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD [
"uvicorn",
"app:app",
"--host",
"0.0.0.0"
]
```

Build:

``` bash
docker build -t bank-model-api .
```

Run:

``` bash
docker run -p 8000:8000 bank-model-api
```

------------------------------------------------------------------------

# 13. GitHub Actions CI Pipeline

.github/workflows/ci.yml

``` yaml
name: CI

on:
  push:
    branches:
      - main

jobs:

  test:

    runs-on: ubuntu-latest

    steps:

    - uses: actions/checkout@v4

    - uses: actions/setup-python@v5
      with:
        python-version: "3.11"

    - run:
        pip install -r requirements.txt

    - run:
        pytest
```

Pipeline:

    Developer
       |
       v
    Git Push
       |
       v
    GitHub Actions
       |
       +--> Install
       |
       +--> Test
       |
       +--> Build

------------------------------------------------------------------------

# 14. Azure Deployment Learning Path

Azure services to learn:

    Azure Storage
          |
          v
    Azure Machine Learning
          |
          v
    Container Registry
          |
          v
    Azure Kubernetes Service
          |
          v
    Monitoring

Useful CLI:

Login:

``` bash
az login
```

Check subscription:

``` bash
az account show
```

Create resource group:

``` bash
az group create \
--name mlops-learning-rg \
--location eastus
```

------------------------------------------------------------------------

# 15. Verification Checklist

Run:

``` bash
make doctor
```

Expected:

    Python 3.11.x
    DVC 3.x.x
    Docker version
    azure-cli
    dagshub OK

    === All checks passed ===

------------------------------------------------------------------------

# 16. Recommended Learning Roadmap

Week 1: - GitHub - Python packaging - ML project structure

Week 2: - DVC - Data pipelines - Reproducible experiments

Week 3: - MLflow - DagsHub - Model registry

Week 4: - Docker - FastAPI - Deployment

Week 5: - Azure ML - Kubernetes - Monitoring

------------------------------------------------------------------------

# Final Goal Project

Build an end-to-end system:

    Dataset
     |
     v
    DVC Versioning
     |
     v
    Training Pipeline
     |
     v
    MLflow Tracking
     |
     v
    DagsHub Registry
     |
     v
    Docker API
     |
     v
    Azure Deployment
     |
     v
    Monitoring
     |
     v
    Retraining

This is the complete MLOps lifecycle.
