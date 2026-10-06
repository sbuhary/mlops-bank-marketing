# Azure Deployment Preparation

This folder documents a practical Azure path for the project.

## Container Registry

1. Build the API image in CI.
2. Push it to Azure Container Registry (ACR).
3. Store registry credentials in GitHub Actions secrets.

## Azure Machine Learning

Use Azure ML for managed training jobs, model registration, and batch validation. The local `params.yaml` can become the source of job parameters.

## Azure Kubernetes Service

Use AKS when the API needs autoscaling, controlled rollout, and production ingress. Deploy the Docker image from ACR and mount secrets through Kubernetes secrets or Azure Key Vault.

## Suggested flow

1. `dvc repro` prepares data and trains the model.
2. MLflow stores run metadata and artifacts.
3. CI builds the API image.
4. ACR stores the image.
5. AKS deploys the image with model artifacts fetched from a registry or mounted storage.
