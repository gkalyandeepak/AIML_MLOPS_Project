import numpy as np
import pandas as pd
import pytest

from src.features import RAW_FEATURES


@pytest.fixture
def sample_df():
    """Small synthetic frame with the same schema as the cleaned data (with NaNs)."""
    rng = np.random.default_rng(0)
    n = 80
    df = pd.DataFrame({
        "age": rng.integers(30, 77, n), "sex": rng.integers(0, 2, n),
        "cp": rng.integers(1, 5, n), "trestbps": rng.integers(94, 200, n),
        "chol": rng.integers(126, 400, n), "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n), "thalach": rng.integers(71, 202, n),
        "exang": rng.integers(0, 2, n), "oldpeak": rng.uniform(0, 6, n).round(1),
        "slope": rng.integers(1, 4, n), "ca": rng.integers(0, 4, n).astype(float),
        "thal": rng.choice([3.0, 6.0, 7.0], n),
    })
    df.loc[0, "ca"] = np.nan
    df.loc[1, "thal"] = np.nan
    df["target"] = (df["oldpeak"] + df["ca"].fillna(0) > 2.5).astype(int)
    return df


@pytest.fixture
def X(sample_df):
    return sample_df[RAW_FEATURES].astype(float)


@pytest.fixture
def y(sample_df):
    return sample_df["target"]
