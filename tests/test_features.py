import numpy as np

from src.features import add_features


def test_hr_reserve_formula(X):
    out = add_features(X)
    expected = X["thalach"] / (220 - X["age"])
    assert np.allclose(out["hr_reserve_pct"], expected)


def test_oldpeak_flag_is_binary(X):
    flag = add_features(X)["oldpeak_flag"]
    assert set(flag.unique()) <= {0, 1}
    assert (flag == (X["oldpeak"] > 0).astype(int)).all()


def test_interaction_feature(X):
    out = add_features(X)
    assert np.allclose(out["exang_x_oldpeak"], X["exang"] * X["oldpeak"])


def test_add_features_does_not_mutate_input(X):
    cols_before = list(X.columns)
    add_features(X)
    assert list(X.columns) == cols_before
