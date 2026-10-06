# Model Registry Workflow

Registered model name: `bank-marketing-model`

MLflow model stages:

- `None`: newly registered model version, not approved yet
- `Staging`: candidate version validated against offline metrics and API smoke tests
- `Production`: version approved for serving

Suggested workflow:

1. Train models with `python -m src.models.train`.
2. Inspect MLflow metrics and artifacts.
3. Register the best run with `python -m src.models.register`.
4. Promote the version to `Staging` in the MLflow UI.
5. Run API and business validation checks.
6. Promote to `Production` after approval.

Secrets are never stored in code. For DagsHub-backed MLflow, set:

```bash
export DAGSHUB_USERNAME=<your-user>
export DAGSHUB_TOKEN=<your-token>
```
