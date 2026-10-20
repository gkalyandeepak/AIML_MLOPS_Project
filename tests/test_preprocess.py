import numpy as np
import pandas as pd

from src.preprocess import FEATURES, TARGET, clean_data


def test_target_is_binary(sample_df):
    raw = sample_df.copy()
    raw[TARGET] = [0, 1, 2, 3, 4] * 16
    out = clean_data(raw)
    assert set(out[TARGET].unique()) <= {0, 1}


def test_target_binarisation_values():
    row = {f: 1 for f in FEATURES}
    raw = pd.DataFrame([{**row, TARGET: t, "age": 40 + t} for t in [0, 1, 4]])
    assert clean_data(raw)[TARGET].tolist() == [0, 1, 1]


def test_question_marks_become_nan(sample_df):
    raw = sample_df.astype(object)
    raw.loc[5, "ca"] = "?"
    out = clean_data(raw)
    assert out["ca"].isna().sum() >= 1
    assert out["ca"].dtype.kind == "f"


def test_duplicates_removed(sample_df):
    doubled = pd.concat([sample_df, sample_df.iloc[:5]], ignore_index=True)
    assert len(clean_data(doubled)) == len(clean_data(sample_df))


def test_missing_target_rows_dropped(sample_df):
    raw = sample_df.copy()
    raw.loc[2, TARGET] = np.nan
    assert len(clean_data(raw)) == len(clean_data(sample_df)) - 1


def test_output_columns(sample_df):
    assert list(clean_data(sample_df).columns) == FEATURES + [TARGET]
