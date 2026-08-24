"""
Fit PCA on the training-split embeddings and check how much variance is
retained at d=64 vs d=128, to pick the working dimension per the report
(Section 3: smallest d that preserves >= 95% variance).

Run this AFTER extract_embeddings.py has produced train_embeddings.npy.
Run: python pca_analysis.py
"""

import os
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import joblib

from config import OUTPUT_DIR, PCA_CANDIDATE_DIMS, VARIANCE_THRESHOLD


def main():
    train_path = os.path.join(OUTPUT_DIR, "train_embeddings.npy")
    train_embeddings = np.load(train_path)
    print(f"Loaded train embeddings: {train_embeddings.shape}")

    # Fit PCA with all components first, just to build the variance curve.
    max_components = min(train_embeddings.shape)
    pca_full = PCA(n_components=max_components)
    pca_full.fit(train_embeddings)

    cum_variance = np.cumsum(pca_full.explained_variance_ratio_)

    # Smallest d that clears the threshold, per the report's rule.
    dims_needed = int(np.argmax(cum_variance >= VARIANCE_THRESHOLD) + 1)
    print(f"Smallest d for >= {VARIANCE_THRESHOLD * 100:.0f}% variance: {dims_needed}")

    print("\nVariance retained at candidate dimensions:")
    for d in PCA_CANDIDATE_DIMS:
        retained = cum_variance[d - 1]
        flag = "OK (>=95%)" if retained >= VARIANCE_THRESHOLD else "below 95%"
        print(f"  d = {d:>3}: {retained * 100:.2f}%  [{flag}]")

    # Plot the curve so you can see it, not just read numbers.
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cum_variance) + 1), cum_variance, label="cumulative variance")
    plt.axhline(VARIANCE_THRESHOLD, color="red", linestyle="--",
                label=f"{int(VARIANCE_THRESHOLD * 100)}% threshold")
    for d in PCA_CANDIDATE_DIMS:
        plt.axvline(d, color="gray", linestyle=":", label=f"d = {d}")
    plt.xlabel("Number of PCA components")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA Variance Retention — Market-1501 Train Embeddings")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(OUTPUT_DIR, "pca_variance_curve.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved variance curve to: {plot_path}")

    # Pick the working dimension: smallest candidate (64 or 128) that clears 95%.
    chosen_d = None
    for d in sorted(PCA_CANDIDATE_DIMS):
        if cum_variance[d - 1] >= VARIANCE_THRESHOLD:
            chosen_d = d
            break
    if chosen_d is None:
        chosen_d = max(PCA_CANDIDATE_DIMS)
        print(f"\nNeither candidate reaches {VARIANCE_THRESHOLD*100:.0f}% variance; "
              f"defaulting to the larger candidate d = {chosen_d}. "
              f"Consider raising d or accepting the shortfall.")
    print(f"\n==> Recommended working dimension: d = {chosen_d}")

    # Fit and save the final PCA + reduced train embeddings at the chosen d.
    pca_final = PCA(n_components=chosen_d)
    reduced_train = pca_final.fit_transform(train_embeddings)

    np.save(os.path.join(OUTPUT_DIR, f"train_embeddings_pca{chosen_d}.npy"), reduced_train)
    joblib.dump(pca_final, os.path.join(OUTPUT_DIR, f"pca_model_d{chosen_d}.joblib"))
    print(f"Saved PCA-reduced train embeddings: {reduced_train.shape}")
    print(f"Saved fitted PCA model (reuse this to transform query/gallery later).")


if __name__ == "__main__":
    main()