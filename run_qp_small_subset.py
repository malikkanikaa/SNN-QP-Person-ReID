"""
Build Q, c, A, b on a SMALL subset of the mined pairs, for the Week 3-4
step of "implement and profile CVXPY and SciPy baselines on small triplet
subsets" (report Section 5). Full-scale QP construction is a Week 7-8 task.

Run this AFTER hard_negative_mining.py.
Run: python run_qp_small_subset.py
"""

import os
import numpy as np

from config import (
    OUTPUT_DIR, PCA_WORKING_DIM, GAMMA_REG, LAMBDA_REG, MARGIN,
    SMALL_SUBSET_SIZE, RANDOM_SEED,
)
from qp_cons import build_diagonal_qp


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    embeddings = np.load(os.path.join(OUTPUT_DIR, f"train_embeddings_pca{PCA_WORKING_DIM}.npy"))
    positive_pairs = np.load(os.path.join(OUTPUT_DIR, "positive_pairs.npy"))
    negative_pairs = np.load(os.path.join(OUTPUT_DIR, "negative_pairs.npy"))

    n_pos = min(SMALL_SUBSET_SIZE, len(positive_pairs))
    n_neg = min(SMALL_SUBSET_SIZE, len(negative_pairs))

    pos_subset = positive_pairs[rng.choice(len(positive_pairs), size=n_pos, replace=False)]
    neg_subset = negative_pairs[rng.choice(len(negative_pairs), size=n_neg, replace=False)]

    print(f"Using subset: {n_pos} positive pairs, {n_neg} negative pairs "
          f"(d = {embeddings.shape[1]})")

    Q, c, A, b, d, n_neg_used = build_diagonal_qp(
        embeddings, pos_subset, neg_subset, GAMMA_REG, LAMBDA_REG, MARGIN,
    )

    print(f"QP size: {Q.shape[0]} variables ({d} weights + {n_neg_used} slacks), "
          f"{A.shape[0]} constraints")

    np.save(os.path.join(OUTPUT_DIR, "qp_Q.npy"), Q)
    np.save(os.path.join(OUTPUT_DIR, "qp_c.npy"), c)
    np.save(os.path.join(OUTPUT_DIR, "qp_A.npy"), A)
    np.save(os.path.join(OUTPUT_DIR, "qp_b.npy"), b)

    print("Saved qp_Q.npy, qp_c.npy, qp_A.npy, qp_b.npy to outputs/ "
          "- ready for solve_cvxpy.py and solve_scipy.py")


if __name__ == "__main__":
    main()