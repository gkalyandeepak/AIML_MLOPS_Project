"""Feature engineering + full preprocessing pipeline (shared by training and API)."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler

BASE_NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
ORDINAL = ["ca"]                       # count of vessels (0-3): monotonic effect
BINARY = ["sex", "fbs", "exang"]
ONEHOT = ["cp", "restecg", "slope", "thal"]  # nominal codes
ENGINEERED_NUMERIC = ["hr_reserve_pct", "exang_x_oldpeak"]
ENGINEERED_BINARY = ["oldpeak_flag"]
RAW_FEATURES = BASE_NUMERIC + ORDINAL + BINARY + ONEHOT


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Domain features derived from the EDA."""
    out = df.copy()
    # achieved max heart rate relative to the age-predicted maximum (220 - age)
    out["hr_reserve_pct"] = out["thalach"] / (220 - out["age"])
    # spike at zero in oldpeak: any ST depression at all
    out["oldpeak_flag"] = (out["oldpeak"] > 0).astype(int)
    # interaction of two strong signals
    out["exang_x_oldpeak"] = out["exang"] * out["oldpeak"]
    return out


def build_pipeline(estimator, engineered: bool = True) -> Pipeline:
    num = BASE_NUMERIC + (ENGINEERED_NUMERIC if engineered else [])
    binary = BINARY + (ENGINEERED_BINARY if engineered else [])
    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", RobustScaler())]), num),   # robust to chol=564
        ("ord", SimpleImputer(strategy="most_frequent"), ORDINAL),
        ("bin", "passthrough", binary),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore",
                                                   sparse_output=False))]), ONEHOT),
    ])
    steps = []
    if engineered:
        steps.append(("features", FunctionTransformer(add_features)))
    steps += [("prep", prep), ("model", estimator)]
    return Pipeline(steps)
