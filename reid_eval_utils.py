"""
Standard Market-1501 evaluation: Euclidean distance, single-query setting,
with junk-image and same-identity/same-camera removal per the official
evaluation protocol (Zheng et al., 2015).

This is the metric implementation shared by dimension_eval.py - it doesn't
touch PCA or embeddings on its own, just scores a given (query, gallery)
feature pair.
"""

import numpy as np
from scipy.spatial.distance import cdist


def evaluate_market1501(query_feats, query_ids, query_cams,
                         gallery_feats, gallery_ids, gallery_cams,
                         ranks=(1, 5, 10, 20),
                         return_dict=False):
    """
    Standard Market-1501 evaluation: Euclidean / Mahalanobis distance, single-query
    setting with junk-image (-1) and same-identity + same-camera removal
    per official protocol (Zheng et al., 2015).

    Parameters:
        query_feats / gallery_feats : (N, d) arrays (raw or metric-transformed).
        query_ids / gallery_ids     : person identity labels.
        query_cams / gallery_cams   : camera IDs.
        ranks                       : tuple/list of rank cutoffs to compute (e.g. 1, 5, 10, 20).
        return_dict                 : if True, returns dict with all metrics;
                                      if False, returns (rank1, mAP, num_valid_queries) for backwards compatibility.

    Returns:
        If return_dict=False:
            (rank1, mAP, num_valid_queries)
        If return_dict=True:
            {
                "rank1": float,
                "rank5": float,
                "rank10": float,
                "rank20": float,
                "ranks": {k: float},
                "mAP": float,
                "num_valid_queries": int,
                "cmc": np.ndarray (first max(ranks) ranks)
            }
    """
    query_ids = np.asarray(query_ids)
    query_cams = np.asarray(query_cams)
    gallery_ids = np.asarray(gallery_ids)
    gallery_cams = np.asarray(gallery_cams)

    dist_mat = cdist(query_feats, gallery_feats, metric="euclidean")

    max_rank = max(ranks) if len(ranks) > 0 else 1
    cmc_hits = np.zeros(max_rank, dtype=np.float64)
    num_valid_queries = 0
    ap_sum = 0.0

    for i in range(dist_mat.shape[0]):
        q_id = query_ids[i]
        q_cam = query_cams[i]

        if q_id <= 0:
            continue  # skip junk/background query

        order = np.argsort(dist_mat[i])
        g_ids_sorted = gallery_ids[order]
        g_cams_sorted = gallery_cams[order]

        # Drop junk gallery entries (-1) and same-identity + same-camera entries
        remove = (g_ids_sorted == -1) | ((g_ids_sorted == q_id) & (g_cams_sorted == q_cam))
        keep = ~remove

        g_ids_valid = g_ids_sorted[keep]
        matches = (g_ids_valid == q_id).astype(np.int32)

        num_rel = matches.sum()
        if num_rel == 0:
            continue  # no valid ground-truth match for this query

        num_valid_queries += 1

        # Cumulative matches along ranking
        cum_hits = np.cumsum(matches)

        # CMC calculation: 1 if first match occurs at or before rank k
        first_match_idx = np.where(matches == 1)[0]
        if len(first_match_idx) > 0:
            first_idx = first_match_idx[0]
            if first_idx < max_rank:
                cmc_hits[first_idx:] += 1.0

        # AP calculation
        precision_at_k = cum_hits / (np.arange(len(matches)) + 1)
        ap = (precision_at_k * matches).sum() / num_rel
        ap_sum += ap

    if num_valid_queries == 0:
        raise ValueError("No valid queries found with positive gallery matches.")

    cmc = cmc_hits / num_valid_queries
    mAP = ap_sum / num_valid_queries
    rank1 = cmc[0]

    if not return_dict:
        return rank1, mAP, num_valid_queries

    rank_dict = {f"rank{k}": float(cmc[k - 1]) for k in ranks if k <= len(cmc)}
    return {
        **rank_dict,
        "ranks": {k: float(cmc[k - 1]) for k in ranks if k <= len(cmc)},
        "rank1": float(rank1),
        "mAP": float(mAP),
        "num_valid_queries": int(num_valid_queries),
        "cmc": cmc
    }