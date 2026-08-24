"""
Apply the already-fitted PCA(d=128) model (pca_model_d128.joblib, from
pca_analysis.py) to the query and gallery embeddings, and save the
reduced versions to disk.

Not required before hard_negative_mining.py (that script only needs
train_embeddings_pca128.npy, which you already have). This is needed
later, when evaluating Rank-1/mAP with the learned metric M - that has
to run in the same 128-D space the QP was solved in.

Run: python apply_pca_to_query_gallery.py
"""

import os
import numpy as np
import joblib

from config import OUTPUT_DIR

PCA_WORKING_DIM = 128

def main():
    pca_path = os.path.join(OUTPUT_DIR, f"pca_model_d{PCA_WORKING_DIM}.joblib")
    pca = joblib.load(pca_path)
    print(f"Loaded PCA model: {pca.n_components_} components")

    for split in ["query", "gallery"]:
        emb_path = os.path.join(OUTPUT_DIR, f"{split}_embeddings.npy")
        embeddings = np.load(emb_path)

        reduced = pca.transform(embeddings)
        out_path = os.path.join(OUTPUT_DIR, f"{split}_embeddings_pca{PCA_WORKING_DIM}.npy")
        np.save(out_path, reduced)

        print(f"{split}: {embeddings.shape} -> {reduced.shape}, saved to {out_path}")


if __name__ == "__main__":
    main()