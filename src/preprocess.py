"""Cleaning + reusable preprocessing transformer (shared by training and API)."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAW_PATH = Path("data/raw/heart_disease_raw.csv")
CLEAN_PATH = Path("data/processed/heart_clean.csv")

NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "target"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Type-fix, binarise target (0 = no disease, 1-4 = disease), drop duplicates.

    Missing values (ca, thal) are deliberately KEPT as NaN here and imputed
    inside the model pipeline, so imputation statistics are learned only from
    training folds (no leakage).
    """
    df = df.copy().replace("?", np.nan)
    for col in FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[TARGET]).drop_duplicates().reset_index(drop=True)
    df[TARGET] = (df[TARGET] > 0).astype(int)
    return df[FEATURES + [TARGET]]


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC),
        ("cat", categorical, CATEGORICAL),
    ])


def main() -> None:
    raw = pd.read_csv(RAW_PATH)
    clean = clean_data(raw)
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(CLEAN_PATH, index=False)
    print(f"Raw: {raw.shape} -> Clean: {clean.shape}")
    print("Missing values:\n", clean.isna().sum()[clean.isna().sum() > 0])
    print("Class balance:\n", clean[TARGET].value_counts(normalize=True).round(3))


if __name__ == "__main__":
    main()
