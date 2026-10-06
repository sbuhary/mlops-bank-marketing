from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.validate import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


def build_preprocessor() -> ColumnTransformer:
    """Create the reusable feature pipeline shared by training and inference."""
    numeric_pipeline = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_pipeline = Pipeline(
        steps=[("encoder", OneHotEncoder(handle_unknown="ignore"))]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ]
    )
