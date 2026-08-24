"""
Build positive (same-identity) pairs and mine hard negative pairs on the
chosen d=128 PCA-reduced train embeddings.

Hard-negative mining keeps |N| under control per anchor: for each anchor
we only score against a random candidate pool (NEG_POOL_SIZE), not the
whole train split, then keep the HARD_NEG_K closest different-identity
images. This is what keeps the QP's slack-variable count manageable
(one slack per negative pair - see report Section 2.3).

Run this AFTER pca_analysis.py (needs train_embeddings_pca128.npy).
Run: python hard_negative_mining.py
"""

import os
import numpy as np
import pandas as pd

from config import (
    OUTPUT_DIR, PCA_WORKING_DIM,
    POS_PAIRS_PER_ID, NEG_POOL_SIZE, HARD_NEG_K, RANDOM_SEED,
)


def build_positive_pairs(identities, rng):
    positive_pairs = []
    unique_ids = np.unique(identities)

    for pid in unique_ids:
        idxs = np.where(identities == pid)[0]
        if len(idxs) < 2:
            continue
        idxs = idxs.copy()
        rng.shuffle(idxs)

        n_pairs = min(POS_PAIRS_PER_ID, len(idxs) // 2)
        for k in range(n_pairs):
            positive_pairs.append((idxs[2 * k], idxs[2 * k + 1]))

    return np.array(positive_pairs, dtype=np.int64)


def mine_hard_negatives(embeddings, identities, rng):
    negative_pairs = []
    all_idxs = np.arange(len(identities))

    for anchor_idx in range(len(embeddings)):
        anchor_id = identities[anchor_idx]

        pool = rng.choice(all_idxs, size=min(NEG_POOL_SIZE, len(all_idxs)), replace=False)
        pool = pool[identities[pool] != anchor_id]
        if len(pool) == 0:
            continue

        dists = np.linalg.norm(embeddings[pool] - embeddings[anchor_idx], axis=1)
        hardest = pool[np.argsort(dists)[:HARD_NEG_K]]

        for neg_idx in hardest:
            negative_pairs.append((anchor_idx, neg_idx))

        if anchor_idx % 1000 == 0:
            print(f"  mined negatives for {anchor_idx}/{len(embeddings)} anchors", end="\r")

    return np.array(negative_pairs, dtype=np.int64)


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    emb_path = os.path.join(OUTPUT_DIR, f"train_embeddings_pca{PCA_WORKING_DIM}.npy")
    embeddings = np.load(emb_path)
    meta = pd.read_csv(os.path.join(OUTPUT_DIR, "train_meta.csv"))
    identities = meta["identity"].values

    print(f"Loaded {embeddings.shape[0]} train embeddings at d={embeddings.shape[1]}")

    positive_pairs = build_positive_pairs(identities, rng)
    print(f"Built {len(positive_pairs)} positive pairs "
          f"across {len(np.unique(identities))} identities")

    negative_pairs = mine_hard_negatives(embeddings, identities, rng)
    print(f"\nMined {len(negative_pairs)} hard negative pairs "
          f"({HARD_NEG_K} per anchor)")

    np.save(os.path.join(OUTPUT_DIR, "positive_pairs.npy"), positive_pairs)
    np.save(os.path.join(OUTPUT_DIR, "negative_pairs.npy"), negative_pairs)
    print("Saved positive_pairs.npy and negative_pairs.npy to outputs/")


if __name__ == "__main__":
    main()