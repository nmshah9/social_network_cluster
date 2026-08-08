"""
run_pipeline.py
----------------
End-to-end script version of the analysis:
    load -> clean -> transform -> outlier-cap -> scale -> cluster (KMeans,
    Hierarchical, DBSCAN) -> compare -> save the winning model + clustered
    dataset to disk.

Run from the project root:
    python run_pipeline.py

This is the script the Streamlit app's "Retrain model" button calls, and
it's also exactly what the notebook walks through interactively with plots.
"""

import numpy as np
import joblib

from src import config
from src.data_preprocessing import run_full_preprocessing
from src.clustering import (
    kmeans_elbow_and_silhouette,
    fit_kmeans,
    fit_hierarchical,
    fit_dbscan,
    dbscan_grid_search,
    compare_algorithms,
    pca_2d,
)


def main(k_for_kmeans: int = None, dbscan_eps: float = None, dbscan_min_samples: int = 8):
    print("=" * 70)
    print("STEP 1-4: Load, clean, treat skew/outliers, scale")
    print("=" * 70)
    prep = run_full_preprocessing(save_artifacts=True)
    scaled_df = prep["scaled_df"]
    X = scaled_df.values

    print(f"Final feature matrix shape: {X.shape}")
    print(f"Log-transformed columns (skew fix): {prep['log_cols']}")

    print("\n" + "=" * 70)
    print("STEP 5: K-Means -- choosing k via silhouette")
    print("=" * 70)
    scores_df = kmeans_elbow_and_silhouette(X, k_range=range(2, 9))
    print(scores_df)
    if k_for_kmeans is None:
        k_for_kmeans = int(scores_df.loc[scores_df["silhouette"].idxmax(), "k"])
    print(f"Selected k = {k_for_kmeans}")
    kmeans_model, kmeans_labels = fit_kmeans(X, k_for_kmeans)

    print("\n" + "=" * 70)
    print("STEP 6a: Hierarchical clustering")
    print("=" * 70)
    hier_model, hier_labels = fit_hierarchical(X, k_for_kmeans)

    print("\n" + "=" * 70)
    print("STEP 6b: DBSCAN")
    print("=" * 70)
    if dbscan_eps is None:
        grid = dbscan_grid_search(
            X,
            eps_values=[0.4, 0.6, 0.8, 1.0, 1.2],
            min_samples_values=[5, 10],
        )
        print(grid.head(10))
        best = grid.dropna(subset=["silhouette"]).iloc[0]
        dbscan_eps, dbscan_min_samples = best["eps"], int(best["min_samples"])
    print(f"Selected eps={dbscan_eps}, min_samples={dbscan_min_samples}")
    dbscan_model, dbscan_labels = fit_dbscan(X, dbscan_eps, dbscan_min_samples)

    print("\n" + "=" * 70)
    print("STEP 6c: Comparing silhouette scores across all 3 algorithms")
    print("=" * 70)
    comparison = compare_algorithms(X, kmeans_labels, hier_labels, dbscan_labels)
    print(comparison)

    print("\nBest algorithm by silhouette score:", comparison.iloc[0]["algorithm"])

    # ---- Persist the K-Means model as the production model ----
    # K-Means is chosen as the deployed model because it (a) scales to new,
    # unseen students in O(1) via `.predict()`, unlike Hierarchical/DBSCAN
    # which have no native out-of-sample prediction, and (b) is typically
    # competitive on this dataset's silhouette score. See the notebook for
    # the full head-to-head comparison and reasoning.
    joblib.dump(kmeans_model, config.KMEANS_MODEL_PATH)

    df_final = prep["df_capped"].copy()
    df_final["cluster"] = kmeans_labels
    df_final.to_csv(config.CLUSTERED_DATA_PATH, index=False)

    _, pca_model = pca_2d(X)
    joblib.dump(pca_model, config.PCA_MODEL_PATH)

    print(f"\nSaved: {config.KMEANS_MODEL_PATH}")
    print(f"Saved: {config.CLUSTERED_DATA_PATH}")
    print(f"Saved: {config.PCA_MODEL_PATH}")
    print("\nPipeline complete.")

    return {
        "comparison": comparison,
        "kmeans_model": kmeans_model,
        "kmeans_labels": kmeans_labels,
        "hier_labels": hier_labels,
        "dbscan_labels": dbscan_labels,
        "df_final": df_final,
    }


if __name__ == "__main__":
    main()
