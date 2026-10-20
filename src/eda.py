"""Generate EDA plots into reports/eda/."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif

from src.preprocess import CATEGORICAL, CLEAN_PATH, NUMERIC, TARGET

OUT = Path("reports/eda")
ORDER = ["No disease", "Disease"]
PALETTE = {"No disease": "#4C72B0", "Disease": "#DD8452"}
sns.set_theme(style="whitegrid", context="notebook")


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT / name)


def cat_frame(df: pd.DataFrame, col: str) -> pd.DataFrame:
    d = df.dropna(subset=[col]).copy()
    d[col] = d[col].astype(int)
    return d


def main() -> None:
    df = pd.read_csv(CLEAN_PATH)
    df["diagnosis"] = df[TARGET].map({0: ORDER[0], 1: ORDER[1]})

    # 1. Class balance with percentages
    counts = df["diagnosis"].value_counts().reindex(ORDER)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(ORDER, counts.values, color=[PALETTE[o] for o in ORDER])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 2, f"{v} ({v / len(df):.1%})", ha="center")
    ax.set_ylabel("Patients")
    ax.set_title("Class balance")
    save(fig, "01_class_balance.png")

    # 2. Numeric histograms by diagnosis
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for ax, col in zip(axes.ravel(), NUMERIC):
        sns.histplot(data=df, x=col, hue="diagnosis", hue_order=ORDER,
                     palette=PALETTE, kde=True, bins=25, ax=ax)
    axes.ravel()[-1].axis("off")
    fig.suptitle("Numeric feature distributions by diagnosis")
    save(fig, "02_numeric_histograms.png")

    # 3. Boxplots (wider figure, no label overlap)
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), constrained_layout=True)
    for ax, col in zip(axes, NUMERIC):
        sns.boxplot(data=df, x="diagnosis", y=col, order=ORDER, hue="diagnosis",
                    hue_order=ORDER, palette=PALETTE, legend=False, ax=ax)
        ax.set_xlabel("")
    fig.suptitle("Numeric features by diagnosis")
    save(fig, "03_boxplots.png")

    # 4. Categorical counts by diagnosis
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    for ax, col in zip(axes.ravel(), CATEGORICAL):
        sns.countplot(data=cat_frame(df, col), x=col, hue="diagnosis",
                      hue_order=ORDER, palette=PALETTE, ax=ax)
    fig.suptitle("Categorical features by diagnosis (counts)")
    save(fig, "04_categorical_counts.png")

    # 5. Disease rate per category, with group sizes
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    for ax, col in zip(axes.ravel(), CATEGORICAL):
        g = cat_frame(df, col).groupby(col)[TARGET].agg(["mean", "size"]).reset_index()
        sns.barplot(data=g, x=col, y="mean", color="#DD8452", ax=ax)
        for i, r in g.iterrows():
            ax.text(i, r["mean"] + 0.02, f"n={int(r['size'])}", ha="center", fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Disease rate")
    fig.suptitle("Disease rate by category (n = group size)")
    save(fig, "05_disease_rate_by_category.png")

    # 6. Correlation heatmap (lower triangle). Pearson on nominal codes is indicative only.
    corr = df[NUMERIC + CATEGORICAL + [TARGET]].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation heatmap (Pearson; nominal codes indicative only)")
    save(fig, "06_correlation_heatmap.png")

    # 7. Mutual information with target (valid for categorical features)
    X = df[NUMERIC + CATEGORICAL].copy()
    X = X.fillna(X.median())
    discrete = [c in CATEGORICAL for c in X.columns]
    mi = pd.Series(
        mutual_info_classif(X, df[TARGET], discrete_features=discrete, random_state=42),
        index=X.columns,
    ).sort_values()
    fig, ax = plt.subplots(figsize=(7, 6))
    mi.plot.barh(ax=ax, color="#4C72B0")
    ax.set_xlabel("Mutual information")
    ax.set_title("Feature relevance to target")
    save(fig, "07_mutual_information.png")
    mi.sort_values(ascending=False).round(3).to_csv(OUT / "mutual_information.csv", header=["mi"])

    # Summary stats
    summary = df[NUMERIC + CATEGORICAL + [TARGET]].describe().T.round(2)
    summary.index.name = "feature"
    summary.to_csv(OUT / "summary_stats.csv")
    print(summary)
    print("\nMutual information (desc):\n", mi.sort_values(ascending=False).round(3))


if __name__ == "__main__":
    main()
