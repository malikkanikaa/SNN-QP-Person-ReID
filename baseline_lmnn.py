"""
LMNN baseline (Weinberger et al., 2009) - report Section 4.1 / Wk 05-06:
"Run LMNN and ITML on reduced embeddings; record Rank-1, mAP, and solve time."

LMNN is the direct structural comparator for the diagonal-M SNN-QP
formulation (Section 4.1). This mirrors solve_cvxpy.py / solve_scipy.py:
same working dimension, similarly small subset size, and evaluates with
the official Market-1501 protocol from reid_eval_utils.py (same one
dimension_eval.py uses), so LMNN's Rank-1/mAP are directly comparable to
your d=128 row in dimension_eval_results.csv.

Requires: pip install metric-learn

Run this AFTER dimension_eval.py (needs train/query/gallery PCA-128
embeddings + *_meta.csv identity/camera columns already in outputs/).
Run: python baseline_lmnn.py
"""

import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from config import OUTPUT_DIR, PCA_WORKING_DIM, SMALL_SUBSET_SIZE, RANDOM_SEED
from eval_reid import evaluate_with_metric

try:
    from metric_learn import LMNN
except ImportError as e:
    raise ImportError(
        "metric-learn is required for this baseline: pip install metric-learn"
    ) from e

LMNN_K = 3  # neighbours LMNN pulls together per anchor; needs k+1 images/identity


def load_split(name, d):
    """Mirrors dimension_eval.py's load_split, but for the PCA-{d} arrays
    saved by the QP pipeline (train_embeddings_pca128.npy etc.) rather than
    the raw 512-D ones."""
    embeddings = np.load(os.path.join(OUTPUT_DIR, f"{name}_embeddings_pca{d}.npy"))
    meta = pd.read_csv(os.path.join(OUTPUT_DIR, f"{name}_meta.csv"))
    return embeddings, meta["identity"].values, meta["camera"].values


def subsample_by_identity(ids, target_size, k, rng):
    """LMNN needs every class in the fit set to have >= k+1 members, so we
    can't just take a random SMALL_SUBSET_SIZE rows (Market-1501 has many
    single/few-image identities in a random slice, which is what triggered
    the "not enough class labels for specified k" error). Instead: keep only
    identities with >= k+1 images, shuffle identities, and pull in whole
    identity groups until we've collected roughly `target_size` rows.
    """
    groups = defaultdict(list)
    for idx, pid in enumerate(ids):
        groups[pid].append(idx)

    eligible = [pid for pid, idxs in groups.items() if len(idxs) >= k + 1]
    if not eligible:
        raise ValueError(
            f"No identity in this split has >= {k + 1} images - lower LMNN_K "
            f"or use a split with more images per identity."
        )
    rng.shuffle(eligible)

    selected = []
    for pid in eligible:
        selected.extend(groups[pid])
        if len(selected) >= target_size:
            break

    return np.array(selected)


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    d = PCA_WORKING_DIM

    train_emb, train_ids, _ = load_split("train", d)
    query_emb, query_ids, query_cams = load_split("query", d)
    gallery_emb, gallery_ids, gallery_cams = load_split("gallery", d)

    idx = subsample_by_identity(train_ids, SMALL_SUBSET_SIZE, LMNN_K, rng)
    X_sub, y_sub = train_emb[idx], train_ids[idx]

    print(f"LMNN training on subset: {X_sub.shape[0]} samples, d = {d}, "
          f"{len(np.unique(y_sub))} identities (all with >= {LMNN_K + 1} images)")

    lmnn = LMNN(k=LMNN_K, learn_rate=1e-6)

    start = time.perf_counter()
    lmnn.fit(X_sub, y_sub)
    elapsed = time.perf_counter() - start
    print(f"LMNN fit time: {elapsed:.4f} s")

    L = lmnn.components_  # transformer such that M = L^T L

    rank1, mAP, n_valid = evaluate_with_metric(
        query_emb, query_ids, query_cams,
        gallery_emb, gallery_ids, gallery_cams, L=L,
    )
    print(f"LMNN Rank-1: {rank1 * 100:.2f}%")
    print(f"LMNN mAP:    {mAP * 100:.2f}%  ({n_valid} valid queries)")

    np.save(os.path.join(OUTPUT_DIR, "lmnn_L.npy"), L)
    results = {"solve_time": elapsed, "rank1": rank1, "mAP": mAP}
    np.save(os.path.join(OUTPUT_DIR, "lmnn_results.npy"), results, allow_pickle=True)
    print("Saved lmnn_L.npy and lmnn_results.npy to outputs/")


if __name__ == "__main__":
    main()