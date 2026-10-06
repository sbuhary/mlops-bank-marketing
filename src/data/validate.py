from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "age",
    "job",
    "marital",
    "education",
    "balance",
    "housing",
    "loan",
    "contact",
    "campaign",
    "previous",
    "poutcome",
    "target",
]

NUMERIC_COLUMNS = ["age", "balance", "campaign", "previous"]
CATEGORICAL_COLUMNS = [
    "job",
    "marital",
    "education",
    "housing",
    "loan",
    "contact",
    "poutcome",
]


@dataclass(frozen=True)
class ValidationResult:
    row_count: int
    target_rate: float
    columns: list[str]


def validate_dataframe(df: pd.DataFrame) -> ValidationResult:
    """Validate the contract between raw data, training, and serving."""
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    unexpected_nulls = df[REQUIRED_COLUMNS].isna().sum()
    null_columns = unexpected_nulls[unexpected_nulls > 0].to_dict()
    if null_columns:
        raise ValueError(f"Null values found in required columns: {null_columns}")

    for column in NUMERIC_COLUMNS + ["target"]:
        df[column] = pd.to_numeric(df[column], errors="raise")

    invalid_targets = sorted(set(df["target"].unique()) - {0, 1})
    if invalid_targets:
        raise ValueError(f"Target must be binary 0/1. Found: {invalid_targets}")

    if len(df) < 10:
        raise ValueError("Dataset is too small for a meaningful train/test split.")

    return ValidationResult(
        row_count=len(df),
        target_rate=float(df["target"].mean()),
        columns=list(df.columns),
    )


def validate_csv(path: str | Path) -> ValidationResult:
    df = pd.read_csv(path)
    return validate_dataframe(df)
