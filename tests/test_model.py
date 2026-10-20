import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.features import build_pipeline


@pytest.mark.parametrize("engineered", [False, True])
def test_pipeline_trains_and_outputs_probabilities(X, y, engineered):
    pipe = build_pipeline(LogisticRegression(max_iter=1000), engineered=engineered)
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.all((proba >= 0) & (proba <= 1))
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_pipeline_handles_missing_values(X, y):
    assert X.isna().sum().sum() > 0
    pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)
    assert not np.isnan(pipe.predict_proba(X)).any()


def test_pipeline_is_deterministic(X, y):
    a = build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y).predict_proba(X)
    b = build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y).predict_proba(X)
    assert np.allclose(a, b)


def test_unseen_category_does_not_crash(X, y):
    pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)
    row = X.iloc[[0]].copy()
    row["thal"] = 99.0
    assert pipe.predict_proba(row).shape == (1, 2)
