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
                         gallery_feats, gallery_ids, gallery_cams):
    """
    Returns (rank1, mAP, num_valid_queries).

    query_feats / gallery_feats : (N, d) arrays, any d - PCA-reduced or full.
    query_ids / gallery_ids     : person identity labels.
    query_cams / gallery_cams   : camera IDs (used for same-camera removal).
    """
    query_ids = np.asarray(query_ids)
    query_cams = np.asarray(query_cams)
    gallery_ids = np.asarray(gallery_ids)
    gallery_cams = np.asarray(gallery_cams)

    dist_mat = cdist(query_feats, gallery_feats, metric="euclidean")

    num_valid_queries = 0
    rank1_hits = 0
    ap_sum = 0.0

    for i in range(dist_mat.shape[0]):
        q_id = query_ids[i]
        q_cam = query_cams[i]

        if q_id <= 0:
            continue  # skip junk/background query, shouldn't normally occur

        order = np.argsort(dist_mat[i])
        g_ids_sorted = gallery_ids[order]
        g_cams_sorted = gallery_cams[order]

        # Drop junk gallery entries and same-identity+same-camera entries
        # (the latter are the query's own camera view, not a valid match).
        remove = (g_ids_sorted == -1) | ((g_ids_sorted == q_id) & (g_cams_sorted == q_cam))
        keep = ~remove

        g_ids_valid = g_ids_sorted[keep]
        matches = (g_ids_valid == q_id).astype(np.int32)

        num_rel = matches.sum()
        if num_rel == 0:
            continue  # no valid ground-truth match for this query, skip it

        num_valid_queries += 1
        rank1_hits += matches[0]

        cum_hits = np.cumsum(matches)
        precision_at_k = cum_hits / (np.arange(len(matches)) + 1)
        ap = (precision_at_k * matches).sum() / num_rel
        ap_sum += ap

    rank1 = rank1_hits / num_valid_queries
    mAP = ap_sum / num_valid_queries
    return rank1, mAP, num_valid_queries