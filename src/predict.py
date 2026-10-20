"""Load the saved pipeline and predict on JSON input (also the base for the API)."""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

from src.features import RAW_FEATURES

MODEL_PATH = Path("models/heart_model.joblib")

SAMPLE = {
    "age": 63, "sex": 1, "cp": 4, "trestbps": 145, "chol": 233, "fbs": 1,
    "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
    "ca": 0, "thal": 6,
}


def load_model(path: Path = MODEL_PATH):
    return joblib.load(path)


def predict(model, record: dict) -> dict:
    missing = [f for f in RAW_FEATURES if f not in record]
    if missing:
        raise ValueError(f"missing features: {missing}")
    X = pd.DataFrame([record])[RAW_FEATURES].astype(float)
    proba = float(model.predict_proba(X)[0, 1])
    pred = int(proba >= 0.5)
    return {"prediction": pred,
            "label": "heart disease" if pred else "no heart disease",
            "confidence": round(proba if pred else 1 - proba, 4),
            "probability_disease": round(proba, 4)}


if __name__ == "__main__":
    rec = json.loads(sys.argv[1]) if len(sys.argv) > 1 else SAMPLE
    print(json.dumps(predict(load_model(), rec), indent=2))
