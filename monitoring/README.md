# Monitoring Foundation

The API appends every prediction to `monitoring/prediction_logs.csv` with:

- request timestamp
- model version
- predicted class
- input features

This gives the project a simple audit trail and a starting point for:

- data drift checks: compare recent feature distributions with `data/processed/train.csv`
- model drift checks: join predictions to delayed ground truth and recompute accuracy/F1
- operational monitoring: track request volume, schema errors, and model availability
