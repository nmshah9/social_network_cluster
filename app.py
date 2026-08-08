"""
app.py
------
Streamlit app for the Students' Social Network Profile Clustering project.

Run with:
    streamlit run app.py

Tabs:
    1. Overview       -- dataset summary
    2. EDA            -- distributions, missing values, correlations
    3. Preprocessing  -- skew/outlier/scaling before-after views
    4. Clustering Lab -- fit & compare K-Means / Hierarchical / DBSCAN live
    5. Cluster Profiles-- demographic profiling + trend-over-time
    6. Predict a Student -- score a brand-new student against the saved model
"""

import os
import sys
import subprocess

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src import config, eda
from src.data_preprocessing import (
    load_data,
    treat_missing_values,
    check_skewness,
    treat_skewness,
    detect_outliers_iqr,
    cap_outliers_iqr,
    scale_features,
)
from src.clustering import (
    kmeans_elbow_and_silhouette,
    plot_elbow_and_silhouette,
    fit_kmeans,
    plot_dendrogram,
    fit_hierarchical,
    plot_k_distance,
    fit_dbscan,
    compare_algorithms,
    plot_algorithm_comparison,
    pca_2d,
    plot_clusters_2d,
    build_cluster_profile,
    plot_cluster_theme_heatmap,
    plot_cluster_trend_over_years,
)

st.set_page_config(page_title="Student Social Network Clustering", layout="wide",
                    page_icon="🎓")

# ---------------------------------------------------------------------------
# Banner (shown at the top of every page)
# ---------------------------------------------------------------------------
BANNER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "banner.png")
if os.path.exists(BANNER_PATH):
    st.image(BANNER_PATH, width="stretch")
# ============================================================
# LOAD BANNER
# ============================================================

banner = Image.open("banner.png")

# ============================================================
# DISPLAY BANNER
# ============================================================

st.image(banner, use_container_width=True)

# ---------------------------------------------------------------------------
# Cached data / model loading
# ---------------------------------------------------------------------------
@st.cache_data
def get_raw_data():
    return load_data()


@st.cache_data
def get_clean_data():
    df_raw = get_raw_data()
    df_clean = treat_missing_values(df_raw)
    numeric_cols = ["age", "NumberOffriends"] + config.INTEREST_COLS
    df_transformed, log_cols = treat_skewness(df_clean, numeric_cols)
    df_capped = cap_outliers_iqr(df_transformed, numeric_cols)
    return df_clean, df_transformed, df_capped, log_cols


@st.cache_resource
def get_trained_artifacts():
    """Load the model/scaler saved by run_pipeline.py, if present."""
    artifacts = {}
    for key, path in [
        ("scaler", config.SCALER_PATH),
        ("kmeans", config.KMEANS_MODEL_PATH),
        ("pca", config.PCA_MODEL_PATH),
        ("feature_cols", config.FEATURE_LIST_PATH),
    ]:
        artifacts[key] = joblib.load(path) if os.path.exists(path) else None
    return artifacts


@st.cache_data
def get_clustered_data():
    if os.path.exists(config.CLUSTERED_DATA_PATH):
        return pd.read_csv(config.CLUSTERED_DATA_PATH)
    return None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🎓 Student Clustering")
st.sidebar.markdown(
    "Segmenting high-school students into interest-based personas using "
    "their social-network profile keyword counts."
)
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "EDA", "Preprocessing", "Clustering Lab", "Cluster Profiles",
     "Predict a Student"],
)
st.sidebar.markdown("---")
if os.path.exists(config.KMEANS_MODEL_PATH):
    st.sidebar.success("Trained model found ✅")
else:
    st.sidebar.warning("No trained model yet. Run `python run_pipeline.py` "
                        "first, or use the Clustering Lab tab.")


# ---------------------------------------------------------------------------
# 1. Overview
# ---------------------------------------------------------------------------
if page == "Overview":
    st.title("Students' Social Network Profile Clustering")
    st.markdown(
        """
This app segments ~30,000 high-school students (here: **{n:,}** rows) based
on keyword counts mined from their social-network profiles, spanning
graduation years **2006–2009**. Each student is described by:

- **Demographics**: graduation year, gender, age, number of friends
- **36 interest keywords**: sports, appearance/romance, performing arts,
  religion, shopping/fashion, and risk-behavior words, each a count of how
  many times that word appeared on the student's profile.

Use the sidebar tabs to explore the EDA, see how the data was cleaned and
transformed, compare three different clustering algorithms, inspect the
resulting student personas, and score a brand-new student profile.
        """.format(n=len(get_raw_data()))
    )

    df_raw = get_raw_data()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Students", f"{len(df_raw):,}")
    c2.metric("Grad years", f"{df_raw['gradyear'].nunique()}")
    c3.metric("Interest features", len(config.INTEREST_COLS))
    c4.metric("Missing age (%)", f"{df_raw['age'].isna().mean()*100:.1f}%")

    st.subheader("Raw data sample")
    st.dataframe(df_raw.head(20), use_container_width=True)

    st.subheader("Column reference")
    st.markdown(
        "- **Demographic columns:** " + ", ".join(config.DEMOGRAPHIC_COLS) + "\n"
        "- **Interest themes:**\n" +
        "\n".join(f"  - *{theme}*: {', '.join(cols)}"
                   for theme, cols in config.INTEREST_THEMES.items())
    )


# ---------------------------------------------------------------------------
# 2. EDA
# ---------------------------------------------------------------------------
elif page == "EDA":
    st.title("Exploratory Data Analysis")
    df_raw = get_raw_data()

    st.subheader("Missing values")
    st.pyplot(eda.plot_missing_values(df_raw))
    st.caption(
        "`gender` and `age` are the only columns with missing values. "
        "Interest-word counts and `NumberOffriends` have no NaNs."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Gender distribution")
        st.pyplot(eda.plot_gender_distribution(treat_missing_values(df_raw)))
    with c2:
        st.subheader("Graduation year distribution")
        st.pyplot(eda.plot_grad_year_distribution(df_raw))

    st.subheader("Age distribution")
    st.pyplot(eda.plot_age_distribution(df_raw))

    st.subheader("Top interest keywords (by total mentions)")
    st.pyplot(eda.plot_top_interest_words(df_raw, config.INTEREST_COLS))

    st.subheader("Interest-theme trends across graduation years")
    st.pyplot(eda.plot_theme_trend_over_years(treat_missing_values(df_raw)))

    with st.expander("Show all interest-word distributions"):
        st.pyplot(eda.plot_numeric_distributions(df_raw, config.INTEREST_COLS))

    with st.expander("Show correlation heatmap"):
        st.pyplot(eda.plot_correlation_heatmap(
            treat_missing_values(df_raw), ["age", "NumberOffriends"] + config.INTEREST_COLS
        ))


# ---------------------------------------------------------------------------
# 3. Preprocessing
# ---------------------------------------------------------------------------
elif page == "Preprocessing":
    st.title("Preprocessing: Skewness, Outliers & Scaling")
    df_clean, df_transformed, df_capped, log_cols = get_clean_data()
    numeric_cols = ["age", "NumberOffriends"] + config.INTEREST_COLS

    st.subheader("1. Skewness check")
    skew_before = check_skewness(df_clean, numeric_cols)
    st.dataframe(skew_before.rename("skewness").to_frame(), use_container_width=True,
                 height=250)
    st.markdown(
        f"**{len(log_cols)} of {len(numeric_cols)}** columns had |skew| > 1 and were "
        f"log1p-transformed: `{', '.join(log_cols)}`"
    )

    st.subheader("2. Distributions before vs. after log1p transform")
    sel_col = st.selectbox("Pick a column to inspect", numeric_cols,
                            index=numeric_cols.index("NumberOffriends"))
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Before transform")
        st.pyplot(eda.plot_numeric_distributions(df_clean, [sel_col], ncols=1))
    with c2:
        st.caption("After transform (if it was skewed)")
        st.pyplot(eda.plot_numeric_distributions(df_transformed, [sel_col], ncols=1))

    st.subheader("3. Outlier detection (IQR method)")
    outlier_summary = detect_outliers_iqr(df_transformed, numeric_cols)
    st.dataframe(outlier_summary, use_container_width=True, height=300)

    st.subheader("4. Boxplots before vs. after capping")
    c3, c4 = st.columns(2)
    with c3:
        st.caption("Before capping")
        st.pyplot(eda.plot_boxplots(df_transformed, [sel_col], ncols=1))
    with c4:
        st.caption("After capping to IQR fences")
        st.pyplot(eda.plot_boxplots(df_capped, [sel_col], ncols=1))

    st.subheader("5. Feature scaling")
    scaled_preview, _ = scale_features(df_capped, numeric_cols)
    st.markdown("StandardScaler is applied so every feature contributes equally "
                "to distance-based clustering (mean 0, std 1):")
    st.dataframe(scaled_preview.describe().T[["mean", "std", "min", "max"]].round(3),
                 use_container_width=True, height=250)


# ---------------------------------------------------------------------------
# 4. Clustering Lab
# ---------------------------------------------------------------------------
elif page == "Clustering Lab":
    st.title("Clustering Lab: K-Means vs. Hierarchical vs. DBSCAN")
    st.markdown(
        "This tab lets you fit all three algorithms live on a sample of the "
        "preprocessed data and compare their silhouette scores. For speed in "
        "the app, plots/fits below use a random sample; `run_pipeline.py` "
        "runs the full-data version and saves the production model."
    )

    _, _, df_capped, _ = get_clean_data()
    feature_cols = ["age", "NumberOffriends"] + config.INTEREST_COLS

    sample_size = st.slider("Sample size for this lab (for responsiveness)",
                             500, min(5000, len(df_capped)), 1500, step=250)
    df_sample = df_capped.sample(sample_size, random_state=config.RANDOM_STATE)
    scaled_sample, scaler = scale_features(df_sample, feature_cols)
    X = scaled_sample.values

    tab_km, tab_hier, tab_db, tab_compare = st.tabs(
        ["K-Means", "Hierarchical", "DBSCAN", "Comparison"]
    )

    with tab_km:
        st.subheader("Choosing k: Elbow + Silhouette")
        k_scores = kmeans_elbow_and_silhouette(X, range(2, 9))
        st.pyplot(plot_elbow_and_silhouette(k_scores))
        st.dataframe(k_scores, use_container_width=True)
        k_choice = st.slider("Choose k for K-Means", 2, 8,
                              int(k_scores.loc[k_scores["silhouette"].idxmax(), "k"]))
        km_model, km_labels = fit_kmeans(X, k_choice)
        coords, _ = pca_2d(X)
        st.pyplot(plot_clusters_2d(coords, km_labels, f"K-Means (k={k_choice}) — PCA projection"))
        st.session_state["km_labels"] = km_labels

    with tab_hier:
        st.subheader("Dendrogram")
        st.pyplot(plot_dendrogram(X))
        k_hier = st.slider("Number of clusters to cut the dendrogram into", 2, 8, 3, key="hier_k")
        hier_model, hier_labels = fit_hierarchical(X, k_hier)
        coords, _ = pca_2d(X)
        st.pyplot(plot_clusters_2d(coords, hier_labels,
                                    f"Hierarchical (k={k_hier}) — PCA projection"))
        st.session_state["hier_labels"] = hier_labels

    with tab_db:
        st.subheader("K-distance plot (to help choose eps)")
        min_samples = st.slider("min_samples", 3, 20, 8)
        st.pyplot(plot_k_distance(X, k=min_samples))
        eps = st.slider("eps", 0.1, 3.0, 1.0, step=0.1)
        db_model, db_labels = fit_dbscan(X, eps, min_samples)
        n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
        noise_pct = 100 * np.mean(db_labels == -1)
        st.markdown(f"**Clusters found:** {n_clusters}  |  **Noise points:** {noise_pct:.1f}%")
        coords, _ = pca_2d(X)
        st.pyplot(plot_clusters_2d(coords, db_labels,
                                    f"DBSCAN (eps={eps}, min_samples={min_samples}) — PCA projection"))
        st.session_state["db_labels"] = db_labels

    with tab_compare:
        st.subheader("Silhouette score comparison")
        if all(k in st.session_state for k in ("km_labels", "hier_labels", "db_labels")):
            comparison = compare_algorithms(
                X, st.session_state["km_labels"], st.session_state["hier_labels"],
                st.session_state["db_labels"], sample_size=min(2000, len(X))
            )
            st.pyplot(plot_algorithm_comparison(comparison))
            st.dataframe(comparison, use_container_width=True)
            st.markdown(
                f"**Best on this sample:** {comparison.iloc[0]['algorithm']} "
                f"(silhouette = {comparison.iloc[0]['silhouette']:.3f})"
            )
        else:
            st.info("Visit the K-Means, Hierarchical, and DBSCAN tabs first so "
                     "each algorithm has been fitted at least once.")


# ---------------------------------------------------------------------------
# 5. Cluster Profiles
# ---------------------------------------------------------------------------
elif page == "Cluster Profiles":
    st.title("Cluster Profiles: Demographics & Trends")
    df_clustered = get_clustered_data()

    if df_clustered is None:
        st.warning("No saved clustering found. Run `python run_pipeline.py` "
                     "from the project root first, then reload this app.")
    else:
        st.markdown(f"Showing the production K-Means clustering "
                    f"(**{df_clustered['cluster'].nunique()} segments**, "
                    f"**{len(df_clustered):,}** students).")

        profile, demo, gender_pct = build_cluster_profile(df_clustered)

        st.subheader("Segment sizes & demographics")
        demo_display = demo.copy()
        demo_display["avg_age"] = demo_display["avg_age"].round(2)
        demo_display["avg_friends"] = demo_display["avg_friends"].round(1)
        st.dataframe(demo_display, use_container_width=True)

        st.subheader("Gender mix per cluster (%)")
        st.dataframe(gender_pct, use_container_width=True)

        st.subheader("Interest-theme intensity per cluster")
        st.pyplot(plot_cluster_theme_heatmap(df_clustered))

        st.subheader("Cluster composition trend across graduation years")
        st.pyplot(plot_cluster_trend_over_years(df_clustered))

        st.subheader("Top interest words per cluster")
        cluster_pick = st.selectbox("Choose a cluster", sorted(df_clustered["cluster"].unique()))
        top_words = profile.loc[cluster_pick].sort_values(ascending=False).head(10)
        st.bar_chart(top_words)


# ---------------------------------------------------------------------------
# 6. Predict a Student
# ---------------------------------------------------------------------------
elif page == "Predict a Student":
    st.title("Predict a New Student's Segment")
    artifacts = get_trained_artifacts()

    if artifacts["kmeans"] is None:
        st.warning("No trained model found. Run `python run_pipeline.py` first.")
    else:
        st.markdown("Enter a student's profile below to see which segment they "
                    "belong to, using the saved production K-Means model.")

        with st.form("predict_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                age = st.number_input("Age", 12.0, 22.0, 17.0, step=0.1)
                n_friends = st.number_input("Number of friends", 0, 1000, 30)
            with c2:
                gradyear = st.selectbox("Graduation year", [2006, 2007, 2008, 2009])
                gender = st.selectbox("Gender", ["Female", "Male", "Unknown"])
            with c3:
                st.markdown("&nbsp;")

            st.markdown("**Interest keyword mentions on profile** (0 if not mentioned)")
            interest_values = {}
            cols_per_row = 6
            interest_list = config.INTEREST_COLS
            for i in range(0, len(interest_list), cols_per_row):
                row_cols = st.columns(cols_per_row)
                for j, colname in enumerate(interest_list[i:i + cols_per_row]):
                    with row_cols[j]:
                        interest_values[colname] = st.number_input(
                            colname, 0, 20, 0, key=f"int_{colname}"
                        )

            submitted = st.form_submit_button("Predict segment")

        if submitted:
            feature_cols = artifacts["feature_cols"]
            scaler = artifacts["scaler"]
            kmeans_model = artifacts["kmeans"]

            new_row = {"age": age, "NumberOffriends": n_friends}
            new_row.update(interest_values)
            new_df = pd.DataFrame([new_row])[feature_cols]

            scaled_new, _ = scale_features(new_df, feature_cols, scaler=scaler)
            cluster_pred = int(kmeans_model.predict(scaled_new.values)[0])

            st.success(f"This student is predicted to belong to **Cluster {cluster_pred}**")

            df_clustered = get_clustered_data()
            if df_clustered is not None:
                profile, demo, gender_pct = build_cluster_profile(df_clustered)
                st.subheader(f"What Cluster {cluster_pred} typically looks like")
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Avg. age", f"{demo.loc[cluster_pred, 'avg_age']:.1f}")
                    st.metric("Avg. friends", f"{demo.loc[cluster_pred, 'avg_friends']:.0f}")
                    st.metric("Segment size", f"{demo.loc[cluster_pred, 'n_students']:,}")
                with c2:
                    top_words = profile.loc[cluster_pred].sort_values(ascending=False).head(8)
                    st.bar_chart(top_words)

# postBuild
python run_pipeline.py

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown("""
<div style='text-align: center;'>

### 🌍 Student Socail Network Profile Cluster using Machine Learning

Built with ❤️ using Streamlit | Developed by nmshah9

</div>
""", unsafe_allow_html=True)
