"""
eda.py
------
Reusable exploratory-data-analysis plotting functions. Each function returns
a matplotlib Figure so it can be displayed in a notebook (`fig`) or embedded
in Streamlit (`st.pyplot(fig)`) without duplicating plotting code.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import config

sns.set_style("whitegrid")


def plot_missing_values(df: pd.DataFrame):
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    if len(missing) == 0:
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center")
        ax.axis("off")
    else:
        sns.barplot(x=missing.values, y=missing.index, ax=ax, color="#4C72B0")
        ax.set_xlabel("Number of missing values")
        ax.set_title("Missing values by column")
    fig.tight_layout()
    return fig


def plot_gender_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5, 4))
    order = df["gender"].value_counts().index
    sns.countplot(x="gender", data=df, order=order, ax=ax, palette="Set2")
    ax.set_title("Gender distribution")
    for p in ax.patches:
        ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom")
    fig.tight_layout()
    return fig


def plot_grad_year_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x="gradyear", data=df, ax=ax, palette="Set2",
                  order=sorted(df["gradyear"].unique()))
    ax.set_title("Graduation year distribution")
    fig.tight_layout()
    return fig


def plot_age_distribution(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(df["age"].dropna(), bins=30, kde=True, ax=axes[0], color="#55A868")
    axes[0].set_title("Age distribution")
    sns.boxplot(x=df["age"].dropna(), ax=axes[1], color="#55A868")
    axes[1].set_title("Age boxplot")
    fig.tight_layout()
    return fig


def plot_numeric_distributions(df: pd.DataFrame, cols, ncols: int = 4):
    n = len(cols)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.array(axes).reshape(-1)
    for i, c in enumerate(cols):
        sns.histplot(df[c], bins=20, ax=axes[i], color="#4C72B0")
        axes[i].set_title(c, fontsize=10)
        axes[i].set_xlabel("")
    for j in range(len(cols), len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    return fig


def plot_boxplots(df: pd.DataFrame, cols, ncols: int = 4):
    n = len(cols)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.array(axes).reshape(-1)
    for i, c in enumerate(cols):
        sns.boxplot(x=df[c], ax=axes[i], color="#C44E52")
        axes[i].set_title(c, fontsize=10)
        axes[i].set_xlabel("")
    for j in range(len(cols), len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, cols):
    fig, ax = plt.subplots(figsize=(14, 11))
    corr = df[cols].corr()
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax, square=True,
                cbar_kws={"shrink": 0.7})
    ax.set_title("Correlation heatmap of interest / demographic variables")
    fig.tight_layout()
    return fig


def plot_top_interest_words(df: pd.DataFrame, interest_cols, top_n: int = 15):
    totals = df[interest_cols].sum().sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=totals.values, y=totals.index, ax=ax, color="#8172B2")
    ax.set_title(f"Top {top_n} interest keywords by total mention count")
    ax.set_xlabel("Total mentions across all profiles")
    fig.tight_layout()
    return fig


def plot_theme_trend_over_years(df: pd.DataFrame, theme_cols_map=config.INTEREST_THEMES):
    """Average mentions per theme, per graduation year -- a simple 'trend over time' view."""
    trend = df.groupby("gradyear")[
        [c for cols in theme_cols_map.values() for c in cols]
    ].mean()

    fig, ax = plt.subplots(figsize=(9, 5))
    for theme, cols in theme_cols_map.items():
        ax.plot(trend.index, trend[cols].mean(axis=1), marker="o", label=theme)
    ax.set_xlabel("Graduation year")
    ax.set_ylabel("Average mentions per profile")
    ax.set_title("Interest-theme trends across graduation years")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    fig.tight_layout()
    return fig
