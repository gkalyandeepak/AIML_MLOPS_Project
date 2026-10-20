"""Generate EDA plots into reports/eda/."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.preprocess import CATEGORICAL, CLEAN_PATH, NUMERIC, TARGET

OUT = Path("reports/eda")
sns.set_theme(style="whitegrid", context="notebook")


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT / name)


def main() -> None:
    df = pd.read_csv(CLEAN_PATH)

    # 1. Class balance
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(data=df, x=TARGET, ax=ax, hue=TARGET, legend=False)
    for p in ax.patches:
        ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom")
    ax.set_xticks([0, 1], ["No disease", "Disease"])
    ax.set_title("Class balance")
    save(fig, "01_class_balance.png")

    # 2. Numeric histograms (split by target)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, col in zip(axes.ravel(), NUMERIC):
        sns.histplot(data=df, x=col, hue=TARGET, kde=True, bins=25, ax=ax)
    axes.ravel()[-1].axis("off")
    fig.suptitle("Numeric feature distributions by target")
    save(fig, "02_numeric_histograms.png")

    # 3. Boxplots
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    for ax, col in zip(axes, NUMERIC):
        sns.boxplot(data=df, x=TARGET, y=col, ax=ax, hue=TARGET, legend=False)
    fig.suptitle("Numeric features vs target")
    save(fig, "03_boxplots.png")

    # 4. Categorical counts by target
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, col in zip(axes.ravel(), CATEGORICAL):
        sns.countplot(data=df, x=col, hue=TARGET, ax=ax)
    fig.suptitle("Categorical features by target")
    save(fig, "04_categorical_counts.png")

    # 5. Correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation heatmap")
    save(fig, "05_correlation_heatmap.png")

    # Summary stats for the report
    summary = df.describe().T.round(2)
    summary.to_csv(OUT / "summary_stats.csv")
    print(summary)


if __name__ == "__main__":
    main()