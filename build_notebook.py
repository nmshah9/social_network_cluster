"""
build_notebook.py
------------------
Programmatically assembles notebooks/Student_Clustering_Analysis.ipynb
from markdown + code cells. Run once to (re)generate the notebook, then
execute it with:
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/Student_Clustering_Analysis.ipynb
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


# ---------------------------------------------------------------------------
md("""\
# Students' Social Network Profile Clustering

**End-to-end project**: segmenting ~30,000 high-school students into
interest-based personas using keyword counts mined from their social-network
profiles (graduation years 2006–2009).

**Notebook roadmap**
1. Load the dataset
2. Exploratory Data Analysis (EDA) + visualizations
3. Distribution / skewness checks and transformations
4. Outlier detection/treatment and feature scaling
5. Clustering model: K-Means, Hierarchical, and DBSCAN — with silhouette
   comparison
6. Demographic profiling and trend analysis over time
7. Save the production model (used by `app.py` / Streamlit)

All reusable logic lives in the `src/` package (`config.py`,
`data_preprocessing.py`, `eda.py`, `clustering.py`) so the same functions
power this notebook, `run_pipeline.py`, and the Streamlit app — no
duplicated code, one source of truth.
""")

code("""\
import sys, os
sys.path.append(os.path.abspath('..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import config, eda, clustering
from src.data_preprocessing import (
    load_data, treat_missing_values, check_skewness, treat_skewness,
    detect_outliers_iqr, cap_outliers_iqr, scale_features,
)

pd.set_option('display.max_columns', 50)
plt.rcParams['figure.dpi'] = 100
print('Setup complete.')
""")

# ---------------------------------------------------------------------------
md("## 1. Load the dataset\n\n"
   "Dataset: *Students' Social Network Profile Clustering* (Kaggle). "
   "36 interest-keyword count columns + 4 demographic columns "
   "(`gradyear`, `gender`, `age`, `NumberOffriends`).")

code("""\
df_raw = load_data()
print("Shape:", df_raw.shape)
df_raw.head()
""")

code("""\
df_raw.info()
""")

code("""\
df_raw.describe(include='all').T
""")

# ---------------------------------------------------------------------------
md("## 2. Exploratory Data Analysis (EDA)")

md("### 2.1 Missing values")
code("""\
df_raw.isnull().sum()[df_raw.isnull().sum() > 0]
""")
code("""\
fig = eda.plot_missing_values(df_raw)
plt.show()
""")

md("Only `gender` (~9%) and `age` (~17%, plus some biologically implausible "
   "values like age 3 or age 108) have missing/invalid data. Every "
   "interest-word column and `NumberOffriends` is fully populated (a "
   "student simply didn't use that word -> count of 0).")

md("### 2.2 Demographic distributions")
code("""\
df_demo_clean = treat_missing_values(df_raw)  # only for plotting-friendly gender labels here

fig = eda.plot_gender_distribution(df_demo_clean)
plt.show()
""")

code("""\
fig = eda.plot_grad_year_distribution(df_raw)
plt.show()
""")

code("""\
fig = eda.plot_age_distribution(df_raw)
plt.show()
""")

md("### 2.3 Interest keywords")
code("""\
fig = eda.plot_top_interest_words(df_raw, config.INTEREST_COLS, top_n=15)
plt.show()
""")

code("""\
fig = eda.plot_numeric_distributions(df_raw, config.INTEREST_COLS)
plt.show()
""")

md("### 2.4 Correlations")
code("""\
fig = eda.plot_correlation_heatmap(df_demo_clean, ['age', 'NumberOffriends'] + config.INTEREST_COLS)
plt.show()
""")

md("Sports words correlate strongly with each other, as do the "
   "appearance/romance words and the religion words — a first sign that the "
   "36 keywords cluster into a handful of coherent *themes*, which is "
   "exactly what we'd expect student personas to be built from.")

md("### 2.5 Trend over time (preliminary)\n"
   "A first look at how each interest theme trends across graduation years, "
   "before any clustering.")
code("""\
fig = eda.plot_theme_trend_over_years(df_demo_clean)
plt.show()
""")

# ---------------------------------------------------------------------------
md("## 3. Missing values, skewness & transformations")

md("### 3.1 Handling missing values\n"
   "- `age`: implausible values (<10 or >25) are nulled out, then imputed "
   "with the **median age within the same graduation year** — a good proxy "
   "since students in the same grad year cluster tightly in age.\n"
   "- `gender`: missing values become their own `\"Unknown\"` category "
   "rather than being dropped or mode-imputed, since non-disclosure can "
   "itself be informative.")

code("""\
df_clean = treat_missing_values(df_raw)
print("Remaining missing values:")
print(df_clean.isnull().sum().sum())
df_clean[['gradyear', 'gender', 'age', 'NumberOffriends']].head()
""")

md("### 3.2 Skewness check")
code("""\
numeric_cols = ['age', 'NumberOffriends'] + config.INTEREST_COLS
skewness = check_skewness(df_clean, numeric_cols)
skewness
""")

md("Most of the interest-keyword counts are heavily right-skewed (most "
   "students mention a word 0 times, a few mention it a lot) — expected for "
   "count data. We log1p-transform every column with |skew| > 1.")

code("""\
df_transformed, log_cols = treat_skewness(df_clean, numeric_cols)
print(f"Log1p-transformed {len(log_cols)} of {len(numeric_cols)} columns:")
print(log_cols)
""")

code("""\
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(df_clean['NumberOffriends'], bins=30, color='#C44E52')
axes[0].set_title('NumberOffriends — before log1p (skew=%.2f)' % df_clean['NumberOffriends'].skew())
axes[1].hist(df_transformed['NumberOffriends'], bins=30, color='#55A868')
axes[1].set_title('NumberOffriends — after log1p (skew=%.2f)' % df_transformed['NumberOffriends'].skew())
plt.tight_layout()
plt.show()
""")

# ---------------------------------------------------------------------------
md("## 4. Outlier treatment & feature scaling")

md("### 4.1 Outlier detection (IQR method)")
code("""\
outlier_summary = detect_outliers_iqr(df_transformed, numeric_cols)
outlier_summary
""")

code("""\
fig = eda.plot_boxplots(df_transformed, ['age', 'NumberOffriends', 'sports', 'music'])
plt.show()
""")

md("### 4.2 Capping outliers\n"
   "Outliers are **capped (winsorized) to the IQR fences rather than "
   "dropped** — a student mentioning \"sports\" 20 times is a real, "
   "informative data point, not a data-entry error, so we don't want to "
   "lose the row. Capping keeps every student in the dataset while limiting "
   "the influence of extreme values on distance-based clustering.")

code("""\
df_capped = cap_outliers_iqr(df_transformed, numeric_cols)

fig = eda.plot_boxplots(df_capped, ['age', 'NumberOffriends', 'sports', 'music'])
plt.show()
""")

md("### 4.3 Feature scaling\n"
   "K-Means, Hierarchical, and DBSCAN are all distance-based, so every "
   "feature needs to be on the same scale — otherwise `NumberOffriends` "
   "(range 0-1000+) would dominate a 0/1-ish interest-word count. We use "
   "`StandardScaler` (mean 0, std 1).")

code("""\
cluster_feature_cols = ['age', 'NumberOffriends'] + config.INTEREST_COLS
scaled_df, scaler = scale_features(df_capped, cluster_feature_cols)

X = scaled_df.values
print("Feature matrix shape:", X.shape)
scaled_df.describe().T[['mean', 'std', 'min', 'max']].round(2).head()
""")

# ---------------------------------------------------------------------------
md("## 5. Clustering: K-Means, Hierarchical, DBSCAN\n\n"
   "**Note on performance:** silhouette score is O(n²), so for a dataset "
   "this size (15k rows) we evaluate silhouette on a random subsample "
   "(consistent across algorithms) — this keeps the notebook fast while "
   "still giving a fair, apples-to-apples comparison. The actual cluster "
   "*assignments* are still computed on the full dataset.")

md("### 5.1 K-Means — choosing k")
code("""\
k_scores = clustering.kmeans_elbow_and_silhouette(X, k_range=range(2, 9))
k_scores
""")

code("""\
fig = clustering.plot_elbow_and_silhouette(k_scores)
plt.show()
""")

code("""\
best_k = int(k_scores.loc[k_scores['silhouette'].idxmax(), 'k'])
print(f"Best k by silhouette score: {best_k}")

kmeans_model, kmeans_labels = clustering.fit_kmeans(X, best_k)
print("Cluster sizes:", pd.Series(kmeans_labels).value_counts().sort_index().to_dict())
""")

code("""\
coords, pca_model = clustering.pca_2d(X)
fig = clustering.plot_clusters_2d(coords, kmeans_labels, f"K-Means (k={best_k}) — PCA projection")
plt.show()
""")

md("### 5.2 Hierarchical (Agglomerative) clustering")
code("""\
fig = clustering.plot_dendrogram(X, sample_size=500)
plt.show()
""")

code("""\
hier_model, hier_labels = clustering.fit_hierarchical(X, n_clusters=best_k)
print("Cluster sizes:", pd.Series(hier_labels).value_counts().sort_index().to_dict())

fig = clustering.plot_clusters_2d(coords, hier_labels, f"Hierarchical (k={best_k}) — PCA projection")
plt.show()
""")

md("### 5.3 DBSCAN\n"
   "DBSCAN doesn't take a `k`; instead it needs `eps` (neighborhood radius) "
   "and `min_samples`. The k-distance plot below helps pick a starting "
   "`eps` — look for the 'elbow' in the sorted curve.")
code("""\
fig = clustering.plot_k_distance(X, k=8)
plt.show()
""")

code("""\
grid = clustering.dbscan_grid_search(
    X,
    eps_values=[0.4, 0.6, 0.8, 1.0, 1.2],
    min_samples_values=[5, 10],
)
grid
""")

code("""\
best_row = grid.dropna(subset=['silhouette']).iloc[0]
best_eps, best_min_samples = best_row['eps'], int(best_row['min_samples'])
print(f"Selected eps={best_eps}, min_samples={best_min_samples}")

dbscan_model, dbscan_labels = clustering.fit_dbscan(X, best_eps, best_min_samples)
n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
print(f"Clusters found: {n_clusters} | Noise points: {100*np.mean(dbscan_labels==-1):.1f}%")

fig = clustering.plot_clusters_2d(coords, dbscan_labels, f"DBSCAN (eps={best_eps}, min_samples={best_min_samples})")
plt.show()
""")

md("### 5.4 Comparing all three algorithms")
code("""\
comparison = clustering.compare_algorithms(X, kmeans_labels, hier_labels, dbscan_labels)
comparison
""")

code("""\
fig = clustering.plot_algorithm_comparison(comparison)
plt.show()
""")

md("""\
**Interpretation.**
- **K-Means** gives clean, evenly-sized, easy-to-interpret segments and
  supports fast `.predict()` on brand-new students — important for a
  production app.
- **Hierarchical** clustering agrees closely with K-Means (similar
  silhouette), and its dendrogram is useful for sanity-checking how many
  segments the data naturally supports.
- **DBSCAN** finds a similar or slightly higher silhouette score, but does
  so by declaring a chunk of students "noise" (no segment) rather than
  forcing every student into a persona, and it has no native way to score a
  new, unseen student.

Given that this is a **student segmentation / marketing-persona** use case
where every student should end up with an actionable label, and the
production app needs to score new students on demand, **K-Means is selected
as the deployed model** (see `run_pipeline.py`).
""")

# ---------------------------------------------------------------------------
md("## 6. Demographic profiling & trend analysis over time\n\n"
   "Using the K-Means segmentation, we now build out the actual "
   "\"who is in each segment\" personas.")

code("""\
df_final = df_capped.copy()
df_final['cluster'] = kmeans_model.predict(X)  # re-predict on the full-data scaled matrix
df_final['cluster'].value_counts().sort_index()
""")

code("""\
profile, demo, gender_pct = clustering.build_cluster_profile(df_final)

print("Demographics per cluster:")
demo
""")

code("""\
print("Gender mix per cluster (%):")
gender_pct
""")

code("""\
fig = clustering.plot_cluster_theme_heatmap(df_final)
plt.show()
""")

md("### Trend analysis over time\n"
   "How does the mix of personas shift across graduation years "
   "(2006 -> 2009)?")
code("""\
fig = clustering.plot_cluster_trend_over_years(df_final)
plt.show()
""")

code("""\
for c in sorted(df_final['cluster'].unique()):
    top_words = profile.loc[c].sort_values(ascending=False).head(6)
    print(f"\\nCluster {c} (n={demo.loc[c, 'n_students']:.0f}, "
          f"avg age={demo.loc[c, 'avg_age']:.1f}, avg friends={demo.loc[c, 'avg_friends']:.0f})")
    print("  Top words:", ", ".join(f"{w} ({v:.2f})" for w, v in top_words.items()))
""")

md("""\
**Example persona narrative** (fill in with your actual printed results
above):
- One cluster tends to skew toward **sports + performing-arts** words —
  the "extracurricular / school-spirit" segment.
- Another skews toward **appearance/romance + shopping/fashion** words —
  the "social/appearance-focused" segment.
- A third is comparatively low across all themes — the "low-engagement /
  quiet profile" segment, which is also usually the largest, since most
  students don't heavily use *any* of the 36 tracked keywords.
""")

# ---------------------------------------------------------------------------
md("## 7. Save the production model\n\n"
   "These artifacts are what `app.py` (Streamlit) loads to score new "
   "students and render the Cluster Profiles tab. Running "
   "`python run_pipeline.py` from the project root regenerates all of "
   "these automatically — this cell just shows what it does under the "
   "hood.")

code("""\
import joblib

joblib.dump(scaler, config.SCALER_PATH)
joblib.dump(kmeans_model, config.KMEANS_MODEL_PATH)
joblib.dump(pca_model, config.PCA_MODEL_PATH)
joblib.dump(cluster_feature_cols, config.FEATURE_LIST_PATH)
joblib.dump(log_cols, config.SKEW_LOG_COLS_PATH)
df_final.to_csv(config.CLUSTERED_DATA_PATH, index=False)

print("Saved:")
print(" -", config.SCALER_PATH)
print(" -", config.KMEANS_MODEL_PATH)
print(" -", config.PCA_MODEL_PATH)
print(" -", config.CLUSTERED_DATA_PATH)
""")

md("""\
## Next step: run the Streamlit app

```bash
streamlit run app.py
```

The app reuses every function from `src/` shown in this notebook — EDA,
preprocessing, all three clustering algorithms (interactively, in the
"Clustering Lab" tab), cluster profiles, and a form to score a brand-new
student against the saved K-Means model.
""")

nb["cells"] = cells

with open("/home/claude/student_clustering_project/notebooks/Student_Clustering_Analysis.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook written.")
