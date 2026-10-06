from __future__ import annotations

from src.data.load_data import prepare_data
from src.data.validate import validate_csv
from src.models.evaluate import evaluate
from src.models.train import train


def main() -> None:
    validate_csv("data/raw/bank.csv")
    prepare_data()
    train()
    evaluate()


if __name__ == "__main__":
    main()
