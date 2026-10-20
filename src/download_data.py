"""Download the UCI Heart Disease dataset (Cleveland, id=45) to data/raw/."""
from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/heart_disease_raw.csv")
UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]


def fetch_via_ucimlrepo() -> pd.DataFrame:
    from ucimlrepo import fetch_ucirepo

    ds = fetch_ucirepo(id=45)
    df = pd.concat([ds.data.features, ds.data.targets], axis=1)
    return df.rename(columns={"num": "target"})


def fetch_via_url() -> pd.DataFrame:
    return pd.read_csv(
        UCI_URL, header=None, names=FEATURES + ["target"], na_values="?"
    )


def download(path: Path = RAW_PATH) -> pd.DataFrame:
    try:
        df = fetch_via_ucimlrepo()
        print("Fetched via ucimlrepo")
    except Exception as exc:  # network/package issue -> fall back to direct URL
        print(f"ucimlrepo failed ({exc}); using direct URL")
        df = fetch_via_url()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} rows x {df.shape[1]} cols -> {path}")
    return df


if __name__ == "__main__":
    download()
