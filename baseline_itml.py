"""
ITML baseline (Davis et al., 2007) - report Section 4.1 / Wk 05-06:
"Run LMNN and ITML on reduced embeddings; record Rank-1, mAP, and solve time."

ITML learns a full (not diagonal-restricted) Mahalanobis metric via
Bregman projection - the full-M baseline referenced in Section 4.1, useful
later for quantifying what the diagonal-M restriction (Section 2.4) costs.

Reuses the exact same mined pairs (positive_pairs.npy / negative_pairs.npy)
and PCA-128 train embeddings that feed qp_cons.build_diagonal_qp, and
evaluates with the official Market-1501 protocol from reid_eval_utils.py
so results are comparable to dimension_eval_results.csv and to LMNN.

Requires: pip install metric-learn

Run this AFTER run_qp_small_subset.py (needs positive_pairs.npy /
negative_pairs.npy / train_embeddings_pca128.npy) and dimension_eval.py
(needs query/gallery PCA-128 embeddings + *_meta.csv).
Run: python baseline_itml.py
"""

import os
import time
import numpy as np
import pandas as pd

from config import OUTPUT_DIR, PCA_WORKING_DIM, SMALL_SUBSET_SIZE, RANDOM_SEED
from eval_reid import evaluate_with_metric

try:
    from metric_learn import ITML
except ImportError as e:
    raise ImportError(
        "metric-learn is required for this baseline: pip install metric-learn"
    ) from e


def load_eval_split(name, d):
    embeddings = np.load(os.path.join(OUTPUT_DIR, f"{name}_embeddings_pca{d}.npy"))
    meta = pd.read_csv(os.path.join(OUTPUT_DIR, f"{name}_meta.csv"))
    return embeddings, meta["identity"].values, meta["camera"].values


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    d = PCA_WORKING_DIM

    train_emb = np.load(os.path.join(OUTPUT_DIR, f"train_embeddings_pca{d}.npy"))
    positive_pairs = np.load(os.path.join(OUTPUT_DIR, "positive_pairs.npy"))
    negative_pairs = np.load(os.path.join(OUTPUT_DIR, "negative_pairs.npy"))

    # ITML wants pairs + a similarity flag (+1 same identity, -1 different),
    # so re-use exactly the same mined pairs used to build the QP
    n_pos = min(SMALL_SUBSET_SIZE, len(positive_pairs))
    n_neg = min(SMALL_SUBSET_SIZE, len(negative_pairs))
    pos_sub = positive_pairs[rng.choice(len(positive_pairs), size=n_pos, replace=False)]
    neg_sub = negative_pairs[rng.choice(len(negative_pairs), size=n_neg, replace=False)]

    pairs = np.vstack([pos_sub, neg_sub])
    y = np.concatenate([np.ones(n_pos, dtype=int), -np.ones(n_neg, dtype=int)])

    print(f"ITML training on subset: {n_pos} positive pairs, {n_neg} negative "
          f"pairs, d = {d}")

    # metric-learn's pair-based API takes (pairs_array, y); pairs_array has
    # shape (n_pairs, 2, n_features)
    pairs_features = np.stack(
        [train_emb[pairs[:, 0]], train_emb[pairs[:, 1]]], axis=1
    )

    itml = ITML()

    start = time.perf_counter()
    itml.fit(pairs_features, y)
    elapsed = time.perf_counter() - start
    print(f"ITML fit time: {elapsed:.4f} s")

    L = itml.components_  # transformer such that M = L^T L

    query_emb, query_ids, query_cams = load_eval_split("query", d)
    gallery_emb, gallery_ids, gallery_cams = load_eval_split("gallery", d)

    rank1, mAP, n_valid = evaluate_with_metric(
        query_emb, query_ids, query_cams,
        gallery_emb, gallery_ids, gallery_cams, L=L,
    )
    print(f"ITML Rank-1: {rank1 * 100:.2f}%")
    print(f"ITML mAP:    {mAP * 100:.2f}%  ({n_valid} valid queries)")

    np.save(os.path.join(OUTPUT_DIR, "itml_L.npy"), L)
    results = {"solve_time": elapsed, "rank1": rank1, "mAP": mAP}
    np.save(os.path.join(OUTPUT_DIR, "itml_results.npy"), results, allow_pickle=True)
    print("Saved itml_L.npy and itml_results.npy to outputs/")


if __name__ == "__main__":
    main()