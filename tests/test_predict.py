import pytest
from sklearn.linear_model import LogisticRegression

from src.features import build_pipeline
from src.predict import SAMPLE, predict


@pytest.fixture
def model(X, y):
    return build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)


def test_predict_output_schema(model):
    out = predict(model, SAMPLE)
    assert set(out) == {"prediction", "label", "confidence", "probability_disease"}
    assert out["prediction"] in (0, 1)
    assert out["confidence"] > 1.0
    assert 0.0 <= out["probability_disease"] <= 1.0


def test_predict_rejects_missing_feature(model):
    bad = {k: v for k, v in SAMPLE.items() if k != "age"}
    with pytest.raises(ValueError, match="missing features"):
        predict(model, bad)
