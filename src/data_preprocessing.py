"""
data_preprocessing.py
----------------------
Loading, cleaning, transforming, and scaling the Students' Social Network
Profile dataset.

Pipeline steps implemented here (matching the project brief):
    1. Load the raw CSV.
    2. Handle missing values (age, gender).
    3. Check skewness of numeric columns and log-transform the skewed ones.
    4. Detect and cap outliers (IQR method).
    5. Scale features (StandardScaler) ready for clustering.

Every function is intentionally small and pure (returns a new DataFrame /
object rather than mutating in place) so the same functions can be reused
from the notebook, from `run_pipeline.py`, and from the Streamlit app.
"""

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

from src import config


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
def load_data(path: str = config.RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw Kaggle CSV into a DataFrame."""
    df = pd.read_csv(path)
    # `age` (and occasionally `gradyear`) can be read in as a pandas
    # string/object dtype depending on the pandas backend, since the column
    # mixes real numbers with blank/NA entries. Force it to numeric so
    # downstream comparisons and math work regardless of pandas version.
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["gradyear"] = pd.to_numeric(df["gradyear"], errors="coerce").astype(int)
    df["gender"] = df["gender"].astype(object).where(df["gender"].notna(), None)
    return df


# ---------------------------------------------------------------------------
# 2. Missing value treatment
# ---------------------------------------------------------------------------
def treat_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    - `age`: contains implausible values (e.g. <10 or >25) and many NaNs.
      We first null-out implausible ages, then impute using the median age
      *within the same graduation year* (a sensible proxy, since students in
      the same grad year cluster tightly in age).
    - `gender`: NaNs are recoded as their own category "Unknown" rather than
      dropped or imputed with a mode, since "did not disclose gender" can
      itself be an informative demographic signal.
    """
    df = df.copy()

    # ---- age ----
    df.loc[(df["age"] < 10) | (df["age"] > 25), "age"] = np.nan
    df["age"] = df.groupby("gradyear")["age"].transform(
        lambda s: s.fillna(s.median())
    )
    # fallback in case a whole gradyear group was NaN
    df["age"] = df["age"].fillna(df["age"].median())

    # ---- gender ----
    df["gender"] = df["gender"].fillna("Unknown")
    df["gender"] = df["gender"].replace({"M": "Male", "F": "Female"})

    # ---- NumberOffriends / interest counts: true zeros, not missing ----
    interest_and_friends = config.INTEREST_COLS + ["NumberOffriends"]
    df[interest_and_friends] = df[interest_and_friends].fillna(0)

    return df


# ---------------------------------------------------------------------------
# 3. Skewness check + transform
# ---------------------------------------------------------------------------
def check_skewness(df: pd.DataFrame, cols) -> pd.Series:
    """Return skewness of each column, sorted descending (most skewed first)."""
    return df[cols].skew().sort_values(ascending=False)


def treat_skewness(df: pd.DataFrame, cols, threshold: float = 1.0):
    """
    Apply log1p to columns whose absolute skewness exceeds `threshold`.
    log1p (log(1+x)) is used instead of log(x) because the interest columns
    and NumberOffriends are counts that legitimately contain zeros.

    Returns:
        df_transformed, list_of_columns_that_were_transformed
    """
    df = df.copy()
    skewness = check_skewness(df, cols)
    skewed_cols = skewness[skewness.abs() > threshold].index.tolist()

    for c in skewed_cols:
        df[c] = np.log1p(df[c].clip(lower=0))

    return df, skewed_cols


# ---------------------------------------------------------------------------
# 4. Outlier detection / treatment
# ---------------------------------------------------------------------------
def detect_outliers_iqr(df: pd.DataFrame, cols) -> pd.DataFrame:
    """
    Return a small summary table: for each column, how many rows fall
    outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
    """
    rows = []
    for c in cols:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = ((df[c] < low) | (df[c] > high)).sum()
        rows.append({"column": c, "lower_bound": low, "upper_bound": high,
                      "n_outliers": n_out,
                      "pct_outliers": round(100 * n_out / len(df), 2)})
    return pd.DataFrame(rows).sort_values("n_outliers", ascending=False)


def cap_outliers_iqr(df: pd.DataFrame, cols, factor: float = 1.5) -> pd.DataFrame:
    """
    Winsorize (cap, don't drop) outliers to the IQR fences. Capping is
    preferred over deleting rows here because a student legitimately can
    mention "sports" 20 times in their profile -- that's a real, informative
    data point, not a data-entry error, so we don't want to lose the row.
    """
    df = df.copy()
    for c in cols:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - factor * iqr, q3 + factor * iqr
        df[c] = df[c].clip(lower=low, upper=high)
    return df


# ---------------------------------------------------------------------------
# 5. Scaling
# ---------------------------------------------------------------------------
def scale_features(df: pd.DataFrame, cols, scaler: StandardScaler = None):
    """
    StandardScale the given columns. If `scaler` is None, a new one is fit;
    otherwise the passed-in (already-fitted) scaler is reused -- this is
    what lets the Streamlit app scale a single new student the same way the
    training data was scaled.

    Returns: (scaled_dataframe_of_just_those_cols, fitted_scaler)
    """
    if scaler is None:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df[cols])
    else:
        scaled = scaler.transform(df[cols])
    scaled_df = pd.DataFrame(scaled, columns=cols, index=df.index)
    return scaled_df, scaler


# ---------------------------------------------------------------------------
# Full pipeline (used by run_pipeline.py and the notebook)
# ---------------------------------------------------------------------------
def run_full_preprocessing(save_artifacts: bool = True):
    """
    Executes the whole cleaning -> skew-fix -> outlier-cap -> scale pipeline
    and returns everything downstream code needs.
    """
    df_raw = load_data()
    df_clean = treat_missing_values(df_raw)

    numeric_cols = ["age", "NumberOffriends"] + config.INTEREST_COLS
    df_transformed, log_cols = treat_skewness(df_clean, numeric_cols)

    df_capped = cap_outliers_iqr(df_transformed, numeric_cols)

    cluster_feature_cols = ["age", "NumberOffriends"] + config.INTEREST_COLS
    scaled_df, scaler = scale_features(df_capped, cluster_feature_cols)

    if save_artifacts:
        joblib.dump(scaler, config.SCALER_PATH)
        joblib.dump(cluster_feature_cols, config.FEATURE_LIST_PATH)
        joblib.dump(log_cols, config.SKEW_LOG_COLS_PATH)
        df_capped.to_csv(config.CLEAN_DATA_PATH, index=False)

    return {
        "df_raw": df_raw,
        "df_clean": df_clean,
        "df_capped": df_capped,
        "scaled_df": scaled_df,
        "scaler": scaler,
        "log_cols": log_cols,
        "feature_cols": cluster_feature_cols,
    }


if __name__ == "__main__":
    result = run_full_preprocessing()
    print("Preprocessing complete.")
    print("Rows:", len(result["df_capped"]))
    print("Log-transformed columns:", result["log_cols"])
