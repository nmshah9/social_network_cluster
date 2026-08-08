# Students' Social Network Profile Clustering

An end-to-end machine learning project that segments ~30,000 high-school
students (2006–2009 graduation years) into interest-based personas, using
keyword counts mined from their social-network profiles.

Built to run locally in **VS Code**: a Jupyter notebook for the analysis
walkthrough, a modular Python package for reusable logic, and a **Streamlit
app** for interactive exploration and live predictions.

---

## What's inside

```
student_clustering_project/
├── data/
│   ├── students_social_network.csv     # raw Kaggle dataset (copy)
│   ├── students_clean.csv              # after cleaning/transform/outlier-cap (generated)
│   └── students_clustered.csv          # final data + cluster labels (generated)
├── models/                             # saved scaler / KMeans model / PCA (generated)
├── notebooks/
│   └── Student_Clustering_Analysis.ipynb   # full narrated analysis, already executed
├── src/
│   ├── config.py               # paths + column groupings (single source of truth)
│   ├── data_preprocessing.py   # load, clean, skew-fix, outlier-cap, scale
│   ├── eda.py                  # all EDA plotting functions
│   └── clustering.py           # K-Means / Hierarchical / DBSCAN + profiling
├── run_pipeline.py             # end-to-end script: run this to (re)train the model
├── build_notebook.py           # (re)generates the notebook programmatically
├── app.py                      # Streamlit app
├── requirements.txt
└── README.md
```

The notebook, `run_pipeline.py`, and `app.py` all import from the same
`src/` package — there's no duplicated logic between the "explore it in a
notebook" and "run it as a script/app" versions.

---

## 1. Setup (VS Code)

```bash
# from the project root
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Open the folder in VS Code, select the `venv` interpreter (Command Palette →
*Python: Select Interpreter*), and install the Jupyter extension if you
don't already have it (needed to run the `.ipynb` file inline in VS Code).

> **Dataset**: `data/students_social_network.csv` is already included — it's
> the "Students' Social Network Profile Clustering" dataset. If you want to
> re-download it from Kaggle yourself, replace this file (keep the same
> column names) and everything downstream still works unchanged.

---

## 2. Run the analysis notebook

Open `notebooks/Student_Clustering_Analysis.ipynb` in VS Code and run all
cells (it's already been executed once end-to-end, so you can also just
read through the saved outputs). It walks through, in order:

1. Loading the dataset
2. EDA + visualizations (missing values, distributions, correlations, top
   interest words, theme trends)
3. Skewness checks + log1p transformation of skewed columns
4. Outlier detection (IQR) + capping, and feature scaling (StandardScaler)
5. K-Means (elbow + silhouette), Hierarchical (dendrogram), and DBSCAN
   (k-distance plot + grid search) — with a side-by-side silhouette-score
   comparison
6. Demographic profiling and trend analysis over graduation years
7. Saving the production model artifacts

---

## 3. Run the full pipeline as a script

To (re)generate every artifact the Streamlit app needs, from the project
root:

```bash
python run_pipeline.py
```

This prints the K-Means k-selection table, the DBSCAN grid search, the
3-way silhouette comparison, and writes:
- `models/scaler.joblib`, `models/kmeans_model.joblib`, `models/pca_model.joblib`
- `data/students_clean.csv`, `data/students_clustered.csv`

Takes about 1.5–2 minutes on a typical laptop.

---

## 4. Run the Streamlit app

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

**Tabs:**
| Tab | What it shows |
|---|---|
| **Overview** | Dataset summary, sample rows, column reference |
| **EDA** | Missing values, demographic distributions, top interest words, correlations |
| **Preprocessing** | Skewness before/after, outlier detection table, boxplots before/after capping, scaled feature summary |
| **Clustering Lab** | Fit K-Means / Hierarchical / DBSCAN live on an adjustable sample, tune parameters with sliders, compare silhouette scores |
| **Cluster Profiles** | Segment sizes, demographics, gender mix, interest-theme heatmap, trend across graduation years, top words per segment |
| **Predict a Student** | Enter a new student's profile → get their predicted segment + how it compares to that segment's typical profile |

The **Predict a Student** and **Cluster Profiles** tabs need
`run_pipeline.py` to have been run at least once (so `models/` and
`data/students_clustered.csv` exist) — the sidebar tells you if a trained
model is missing.

---

## Method notes

- **Missing values**: `age` outliers/NaNs imputed by grad-year median;
  `gender` NaNs become an explicit `"Unknown"` category rather than being
  dropped.
- **Skewness**: columns with |skew| > 1 are log1p-transformed (not log, to
  handle the many zero counts in the interest-word columns).
- **Outliers**: capped (winsorized) to IQR fences rather than dropped — an
  unusually high interest-word count is a real signal, not a data-entry
  error.
- **Scaling**: StandardScaler on all clustering features, since K-Means/
  Hierarchical/DBSCAN are distance-based and `NumberOffriends` would
  otherwise dominate the 0/1-ish interest counts.
- **Silhouette scoring at scale**: silhouette score is O(n²), so on the
  full 15k-row dataset it's computed on a consistent random subsample
  across all three algorithms — fast, and still an apples-to-apples
  comparison.
- **Model chosen for production**: K-Means. It was competitive on
  silhouette score with Hierarchical and DBSCAN, but unlike them it (a)
  supports fast `.predict()` on brand-new, unseen students, and (b) assigns
  every student to a segment rather than leaving some as "noise" — both
  important for a usable persona-segmentation tool.
