"""
config.py
---------
Central place for file paths and column groupings used across the project.
Keeping these in one module means the notebook, the pipeline scripts, and the
Streamlit app all agree on the same definitions.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

RAW_DATA_PATH = os.path.join(DATA_DIR, "students_social_network.csv")
CLEAN_DATA_PATH = os.path.join(DATA_DIR, "students_clean.csv")
CLUSTERED_DATA_PATH = os.path.join(DATA_DIR, "students_clustered.csv")

SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
KMEANS_MODEL_PATH = os.path.join(MODELS_DIR, "kmeans_model.joblib")
PCA_MODEL_PATH = os.path.join(MODELS_DIR, "pca_model.joblib")
FEATURE_LIST_PATH = os.path.join(MODELS_DIR, "feature_columns.joblib")
SKEW_LOG_COLS_PATH = os.path.join(MODELS_DIR, "log_transformed_columns.joblib")

for _d in (DATA_DIR, MODELS_DIR, OUTPUTS_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------------
# Column groupings
# ---------------------------------------------------------------------------
# Demographic / profile columns (not "interest word counts")
DEMOGRAPHIC_COLS = ["gradyear", "gender", "age", "NumberOffriends"]

# The 36 interest keywords whose counts were mined from each student's
# social-network profile text. These drive the segmentation.
INTEREST_COLS = [
    "basketball", "football", "soccer", "softball", "volleyball", "swimming",
    "cheerleading", "baseball", "tennis", "sports",
    "cute", "sex", "sexy", "hot", "kissed", "dance",
    "band", "marching", "music", "rock",
    "god", "church", "jesus", "bible",
    "hair", "dress", "blonde", "mall", "shopping", "clothes",
    "hollister", "abercrombie",
    "die", "death", "drunk", "drugs",
]

# Thematic groupings of the interest words, used for demographic / trend
# profiling (radar charts, grouped bar charts, etc.)
INTEREST_THEMES = {
    "Sports": ["basketball", "football", "soccer", "softball", "volleyball",
               "swimming", "cheerleading", "baseball", "tennis", "sports"],
    "Appearance/Romance": ["cute", "sex", "sexy", "hot", "kissed", "hair",
                            "dress", "blonde"],
    "Performing Arts": ["dance", "band", "marching", "music", "rock"],
    "Religion": ["god", "church", "jesus", "bible"],
    "Shopping/Fashion": ["mall", "shopping", "clothes", "hollister",
                          "abercrombie"],
    "Risk Behavior": ["die", "death", "drunk", "drugs"],
}

RANDOM_STATE = 42
