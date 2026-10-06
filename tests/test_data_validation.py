import pandas as pd
import pytest

from src.data.validate import validate_dataframe


def test_validate_dataframe_accepts_bank_schema():
    df = pd.read_csv("data/raw/bank.csv")
    result = validate_dataframe(df)
    assert result.row_count >= 10
    assert 0 <= result.target_rate <= 1


def test_validate_dataframe_rejects_missing_target():
    df = pd.read_csv("data/raw/bank.csv").drop(columns=["target"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_dataframe(df)
