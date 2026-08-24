"""
Compare Rank-1 / mAP at each candidate PCA dimension (config.EVAL_DIMS)
against the full 512-D embedding, using plain Euclidean distance.

Picks the smallest dimension whose mAP is within MAP_TOLERANCE of the
full 512-D mAP - this is the justification number for the report.

Run this AFTER extract_embeddings.py.
Run: python dimension_eval.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from config import OUTPUT_DIR, EVAL_DIMS, MAP_TOLERANCE
from reid_eval_utils import evaluate_market1501


def load_split(name):
    embeddings = np.load(os.path.join(OUTPUT_DIR, f"{name}_embeddings.npy"))
    meta = pd.read_csv(os.path.join(OUTPUT_DIR, f"{name}_meta.csv"))
    return embeddings, meta["identity"].values, meta["camera"].values


def main():
    train_emb, _, _ = load_split("train")
    query_emb, query_ids, query_cams = load_split("query")
    gallery_emb, gallery_ids, gallery_cams = load_split("gallery")

    results = []

    # ---- Full 512-D baseline (no PCA) --------------------------------------
    rank1, mAP, n = evaluate_market1501(
        query_emb, query_ids, query_cams,
        gallery_emb, gallery_ids, gallery_cams,
    )
    print(f"d = 512 (full): Rank-1 = {rank1 * 100:.2f}%, mAP = {mAP * 100:.2f}%  ({n} valid queries)")
    results.append({"dimension": 512, "rank1": rank1, "mAP": mAP})

    # ---- Candidate reduced dimensions ---------------------------------------
    for d in EVAL_DIMS:
        pca = PCA(n_components=d)
        pca.fit(train_emb)  # PCA fit on train split only, applied to query/gallery

        q_reduced = pca.transform(query_emb)
        g_reduced = pca.transform(gallery_emb)

        rank1, mAP, n = evaluate_market1501(
            q_reduced, query_ids, query_cams,
            g_reduced, gallery_ids, gallery_cams,
        )
        print(f"d = {d}: Rank-1 = {rank1 * 100:.2f}%, mAP = {mAP * 100:.2f}%  ({n} valid queries)")
        results.append({"dimension": d, "rank1": rank1, "mAP": mAP})

    results_df = pd.DataFrame(results).sort_values("dimension").reset_index(drop=True)
    results_csv = os.path.join(OUTPUT_DIR, "dimension_eval_results.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"\nSaved results table to {results_csv}")

    # ---- Pick smallest d whose mAP is close to the full-D mAP ---------------
    full_mAP = results_df.loc[results_df["dimension"] == 512, "mAP"].iloc[0]
    candidates = results_df[results_df["dimension"] < 512].copy()
    candidates["mAP_gap"] = full_mAP - candidates["mAP"]

    close_enough = candidates[candidates["mAP_gap"] <= MAP_TOLERANCE]

    if len(close_enough) > 0:
        chosen_d = int(close_enough["dimension"].min())
        print(f"\n==> Recommended working dimension: d = {chosen_d} "
              f"(within {MAP_TOLERANCE * 100:.1f} pts of full-D mAP)")
    else:
        chosen_d = int(candidates.loc[candidates["mAP"].idxmax(), "dimension"])
        print(f"\nNo candidate is within {MAP_TOLERANCE * 100:.1f} pts of full-D mAP; "
              f"falling back to the best-performing candidate: d = {chosen_d}")


if __name__ == "__main__":
    main()