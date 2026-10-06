from src.features.preprocessing import build_preprocessor
from src.models.predict import normalize_features


def test_preprocessor_has_expected_transformers():
    preprocessor = build_preprocessor()
    transformer_names = [name for name, _, _ in preprocessor.transformers]
    assert transformer_names == ["numeric", "categorical"]


def test_normalize_features_fills_serving_defaults():
    frame = normalize_features(
        {
            "age": 40,
            "job": "admin",
            "balance": 5000,
            "housing": "yes",
            "loan": "no",
        }
    )
    assert frame.loc[0, "campaign"] == 2
    assert frame.loc[0, "education"] == "secondary"
