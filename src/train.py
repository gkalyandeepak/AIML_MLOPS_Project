"""Train, tune, evaluate and track models with MLflow.

3 models x 2 feature sets (baseline / engineered) = 6 tracked runs, then the
best configuration (by repeated-CV ROC-AUC) is refit on all data and saved.
"""
import json
import os
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
from mlflow.models import infer_signature
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold, cross_validate, train_test_split)

from src.features import RAW_FEATURES, build_pipeline
from src.preprocess import CLEAN_PATH, TARGET

SEED = 42
EXPERIMENT = "heart-disease-classification"
REGISTERED_NAME = "heart-disease-classifier"
PLOTS = Path("reports/models")
MODEL_PATH = Path("models/heart_model.joblib")
SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]

MODELS = {
    "logistic_regression": (
        LogisticRegression(max_iter=2000, random_state=SEED),
        {"model__C": [0.01, 0.1, 1, 10, 100]},
    ),
    "random_forest": (
        RandomForestClassifier(random_state=SEED, n_jobs=1),
        {"model__n_estimators": [200, 400],
         "model__max_depth": [3, 5, None],
         "model__min_samples_leaf": [1, 3]},
    ),
    "gradient_boosting": (
        GradientBoostingClassifier(random_state=SEED),
        {"model__n_estimators": [100, 200],
         "model__learning_rate": [0.03, 0.1],
         "model__max_depth": [2, 3]},
    ),
}


def load_data():
    df = pd.read_csv(CLEAN_PATH)
    X = df[RAW_FEATURES].astype(float)   # floats everywhere -> stable MLflow schema
    y = df[TARGET].astype(int)
    return X, y


def log_model(model, X, registered_name=None):
    kwargs = dict(serialization_format="cloudpickle", signature=infer_signature(X, model.predict(X)),
                  input_example=X.head(3), registered_model_name=registered_name)
    try:
        return mlflow.sklearn.log_model(model, name="model", **kwargs)
    except TypeError:  # older MLflow
        return mlflow.sklearn.log_model(model, artifact_path="model", **kwargs)


def save_eval_plots(model, y_test, proba, pred, tag):
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y_test, proba, ax=ax, name=tag)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_title(f"ROC curve: {tag}")
    roc_path = PLOTS / f"roc_{tag}.png"
    fig.savefig(roc_path, dpi=150, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=["No disease", "Disease"], ax=ax, colorbar=False)
    ax.set_title(f"Confusion matrix: {tag}")
    cm_path = PLOTS / f"cm_{tag}.png"
    fig.savefig(cm_path, dpi=150, bbox_inches="tight"); plt.close(fig)

    paths = [roc_path, cm_path]
    est = model.named_steps["model"]
    names = model.named_steps["prep"].get_feature_names_out()
    if hasattr(est, "feature_importances_"):
        imp, label = est.feature_importances_, "Feature importance"
    elif hasattr(est, "coef_"):
        imp, label = np.abs(est.coef_[0]), "|Coefficient|"
    else:
        imp = None
    if imp is not None:
        s = pd.Series(imp, index=names).sort_values().tail(15)
        fig, ax = plt.subplots(figsize=(7, 5))
        s.plot.barh(ax=ax, color="#4C72B0")
        ax.set_title(f"{label}: {tag}")
        fi_path = PLOTS / f"importance_{tag}.png"
        fig.savefig(fi_path, dpi=150, bbox_inches="tight"); plt.close(fig)
        paths.append(fi_path)
    return paths


def run_experiment(name, engineered, X_train, X_test, y_train, y_test):
    feature_set = "engineered" if engineered else "baseline"
    tag = f"{name}_{feature_set}"
    estimator, grid = MODELS[name]
    pipe = build_pipeline(clone(estimator), engineered=engineered)

    with mlflow.start_run(run_name=tag):
        mlflow.set_tags({"model": name, "feature_set": feature_set, "stage": "experiment"})
        gs = GridSearchCV(pipe, grid, scoring="roc_auc", n_jobs=-1,
                          cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
        gs.fit(X_train, y_train)
        best = gs.best_estimator_

        # repeated stratified CV (5 folds x 3 repeats) on the training split
        cv = cross_validate(
            clone(best), X_train, y_train, scoring=SCORING, n_jobs=-1,
            cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED))
        cv_metrics = {}
        for m in SCORING:
            cv_metrics[f"cv_{m}_mean"] = float(np.mean(cv[f"test_{m}"]))
            cv_metrics[f"cv_{m}_std"] = float(np.std(cv[f"test_{m}"]))

        # honest estimate on the untouched hold-out set
        proba = best.predict_proba(X_test)[:, 1]
        pred = best.predict(X_test)
        test_metrics = {
            "test_accuracy": accuracy_score(y_test, pred),
            "test_precision": precision_score(y_test, pred),
            "test_recall": recall_score(y_test, pred),
            "test_f1": f1_score(y_test, pred),
            "test_roc_auc": roc_auc_score(y_test, proba),
        }

        mlflow.log_params({"model": name, "feature_set": feature_set, "seed": SEED,
                           "cv": "RepeatedStratifiedKFold(5x3)", "tuning": "GridSearchCV(5-fold)",
                           "test_size": 0.2, "scaler": "RobustScaler",
                           **{k.replace("model__", "best_"): v for k, v in gs.best_params_.items()}})
        mlflow.log_metrics({**cv_metrics, **test_metrics, "tuning_best_roc_auc": gs.best_score_})
        for p in save_eval_plots(best, y_test, proba, pred, tag):
            mlflow.log_artifact(str(p), artifact_path="plots")
        log_model(best, X_train)

    row = {"run": tag, "model": name, "feature_set": feature_set,
           "best_params": json.dumps({k.replace("model__", ""): v
                                      for k, v in gs.best_params_.items()}, default=str),
           **cv_metrics, **test_metrics}
    print(f"{tag:40s} cv_auc={cv_metrics['cv_roc_auc_mean']:.3f} "
          f"test_auc={test_metrics['test_roc_auc']:.3f}")
    return row, best


def main():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(EXPERIMENT)
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    print(f"train={len(X_train)} test={len(X_test)}")

    if Path("reports/eda").exists():
        with mlflow.start_run(run_name="eda_artifacts"):
            mlflow.set_tag("stage", "eda")
            mlflow.log_artifacts("reports/eda", artifact_path="eda")

    rows, fitted = [], {}
    for name in MODELS:
        for engineered in (False, True):
            row, best = run_experiment(name, engineered, X_train, X_test, y_train, y_test)
            rows.append(row)
            fitted[row["run"]] = best

    results = pd.DataFrame(rows).sort_values("cv_roc_auc_mean", ascending=False)
    PLOTS.mkdir(parents=True, exist_ok=True)
    results.to_csv(PLOTS / "model_comparison.csv", index=False)
    print("\n", results[["run", "cv_roc_auc_mean", "cv_roc_auc_std", "cv_accuracy_mean",
                         "cv_precision_mean", "cv_recall_mean", "test_roc_auc"]].round(3).to_string(index=False))

    # final model: best by repeated-CV ROC-AUC, refit on ALL data for deployment
    top = results.iloc[0]
    final = clone(fitted[top["run"]]).fit(X, y)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, MODEL_PATH)
    meta = {"selected_run": top["run"], "model": top["model"], "feature_set": top["feature_set"],
            "best_params": json.loads(top["best_params"]),
            "cv_roc_auc_mean": top["cv_roc_auc_mean"], "test_roc_auc": top["test_roc_auc"],
            "features": RAW_FEATURES, "sklearn_version": sklearn.__version__,
            "trained_on_rows": int(len(X))}
    Path("models/metadata.json").write_text(json.dumps(meta, indent=2, default=float))

    with mlflow.start_run(run_name="final_model"):
        mlflow.set_tags({"stage": "final", "selected_run": top["run"]})
        mlflow.log_params({"selected_model": top["model"], "feature_set": top["feature_set"],
                           "refit_on": "all_data"})
        mlflow.log_metrics({"cv_roc_auc_mean": top["cv_roc_auc_mean"],
                            "holdout_roc_auc": top["test_roc_auc"]})
        mlflow.log_artifact(str(PLOTS / "model_comparison.csv"))
        mlflow.log_artifact("models/metadata.json")
        log_model(final, X, registered_name=REGISTERED_NAME)
    print(f"\nSelected: {top['run']}  -> saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
