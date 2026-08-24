"""
Builds the explicit diagonal-M QP arrays (Q, c, A, b) described in
Section 2.3 / 2.4 of the report.

Variable vector:  z = [ w ; xi ] in R^{d + |N|}   (w = diag(M), see 2.4)

    minimize    0.5 * z^T Q z + c^T z
    subject to  A z <= b

Q = block-diag(2*gamma*I_d, 0_{|N|})
c = [ sum_{(i,j) in P} (x_i - x_j)^2 (elementwise) ; lambda * 1_{|N|} ]

Constraints, one block per type:
  C1 (margin, per negative pair):     -(x_i-x_k)^2 . w - xi_{(i,k)} <= -m
  C2 (slack non-negativity):          -xi_{(i,k)} <= 0
  C3 (diagonal-PSD box constraint):   -w_r <= 0   for each dimension r
"""

import numpy as np


def build_diagonal_qp(embeddings, positive_pairs, negative_pairs, gamma, lam, margin):
    d = embeddings.shape[1]
    n_neg = len(negative_pairs)
    n_vars = d + n_neg

    # ---- Q ----
    Q = np.zeros((n_vars, n_vars))
    Q[:d, :d] = 2 * gamma * np.eye(d)

    # ---- c ----
    c_w = np.zeros(d)
    for i, j in positive_pairs:
        diff = embeddings[i] - embeddings[j]
        c_w += diff ** 2
    c = np.concatenate([c_w, lam * np.ones(n_neg)])

    # ---- C1: margin constraint, one row per negative pair ----
    rows_C1 = np.zeros((n_neg, n_vars))
    b_C1 = np.full(n_neg, -margin)
    for row, (i, k) in enumerate(negative_pairs):
        diff_sq = (embeddings[i] - embeddings[k]) ** 2
        rows_C1[row, :d] = -diff_sq
        rows_C1[row, d + row] = -1.0

    # ---- C2: slack non-negativity ----
    rows_C2 = np.zeros((n_neg, n_vars))
    for row in range(n_neg):
        rows_C2[row, d + row] = -1.0
    b_C2 = np.zeros(n_neg)

    # ---- C3: box constraint replacing the PSD cone (w_r >= 0) ----
    rows_C3 = np.zeros((d, n_vars))
    for row in range(d):
        rows_C3[row, row] = -1.0
    b_C3 = np.zeros(d)

    A = np.vstack([rows_C1, rows_C2, rows_C3])
    b = np.concatenate([b_C1, b_C2, b_C3])

    return Q, c, A, b, d, n_neg


# NOTE for Week 7-8 (full-scale run): this builds dense A/Q, fine for the
# small subsets used here to profile/verify solvers. At full scale
# (d=128, |N| in the tens of thousands) rebuild A and Q with
# scipy.sparse (A is extremely sparse - each C1/C2 row has only 2
# nonzeros) before feeding OSQP/CVXPY, or dense memory use will blow up.