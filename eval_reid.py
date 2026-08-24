"""
Thin wrapper around reid_eval_utils.evaluate_market1501 that lets LMNN/ITML
apply their learned transform L (M = L^T L) before scoring, so we reuse the
same official Market-1501 protocol (junk removal, same-identity/same-camera
removal) as dimension_eval.py rather than re-implementing it.

Kept as a separate module (rather than editing reid_eval_utils.py) since
that file is explicitly "the metric implementation shared by
dimension_eval.py" - baselines import from it, they don't modify it.
"""

import numpy as np

from reid_eval_utils import evaluate_market1501


def evaluate_with_metric(query_feats, query_ids, query_cams,
                          gallery_feats, gallery_ids, gallery_cams, L=None):
    """Rank-1 / mAP / n_valid_queries after projecting features by L.

    L=None -> plain Euclidean (identical to dimension_eval.py's baseline).
    L = metric_learn's `.components_` -> Mahalanobis distance under the
    learned metric M = L^T L, since ||L x - L y||^2 = (x-y)^T M (x-y).
    """
    if L is not None:
        query_feats = query_feats @ L.T
        gallery_feats = gallery_feats @ L.T
    return evaluate_market1501(
        query_feats, query_ids, query_cams,
        gallery_feats, gallery_ids, gallery_cams,
    )