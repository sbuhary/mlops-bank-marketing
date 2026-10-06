from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.data.validate import validate_dataframe


def load_params(params_path: str | Path = "params.yaml") -> dict:
    with Path(params_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_raw_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    validate_dataframe(df)
    return df


def prepare_data(params_path: str | Path = "params.yaml") -> tuple[pd.DataFrame, pd.DataFrame]:
    params = load_params(params_path)
    data_params = params["data"]
    split_params = params["split"]

    df = load_raw_data(data_params["raw_path"])
    train_df, test_df = train_test_split(
        df,
        test_size=split_params["test_size"],
        random_state=split_params["random_state"],
        stratify=df[data_params["target"]],
    )

    processed_dir = Path(data_params["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(processed_dir / "train.csv", index=False)
    test_df.to_csv(processed_dir / "test.csv", index=False)
    return train_df, test_df


def main() -> None:
    train_df, test_df = prepare_data()
    print(f"Prepared data: train={len(train_df)} rows, test={len(test_df)} rows")


if __name__ == "__main__":
    main()
