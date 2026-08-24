from snn_qp_solver_new import diagnose_diagonal_qp, tune_diagonal_qp_snn
import numpy as np

Q = np.load("outputs/qp_Q.npy")
c = np.load("outputs/qp_c.npy")
A = np.load("outputs/qp_A.npy")
b = np.load("outputs/qp_b.npy")

d = np.count_nonzero(np.diag(Q))
n_neg = Q.shape[0] - d

diagnose_diagonal_qp(Q, c, A, b, d, n_neg)
tune_diagonal_qp_snn(Q, c, A, b, d, n_neg, max_iters=3000)