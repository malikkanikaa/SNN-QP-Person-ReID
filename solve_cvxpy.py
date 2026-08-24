"""
Solve the diagonal-M QP with CVXPY (OSQP backend) - the "ground truth"
numerical baseline referenced in report Section 4.2/4.3.

Run this AFTER run_qp_small_subset.py.
Run: python solve_cvxpy.py
"""

import os
import time
import numpy as np
import cvxpy as cp

from config import OUTPUT_DIR


def main():
    Q = np.load(os.path.join(OUTPUT_DIR, "qp_Q.npy"))
    c = np.load(os.path.join(OUTPUT_DIR, "qp_c.npy"))
    A = np.load(os.path.join(OUTPUT_DIR, "qp_A.npy"))
    b = np.load(os.path.join(OUTPUT_DIR, "qp_b.npy"))

    n_vars = Q.shape[0]
    z = cp.Variable(n_vars)

    objective = cp.Minimize(0.5 * cp.quad_form(z, Q) + c @ z)
    constraints = [A @ z <= b]
    problem = cp.Problem(objective, constraints)

    start = time.perf_counter()
    problem.solve(solver=cp.OSQP)
    elapsed = time.perf_counter() - start

    print(f"CVXPY (OSQP) status: {problem.status}")
    print(f"Objective value: {problem.value:.6f}")
    print(f"Solve time: {elapsed:.4f} s")

    np.save(os.path.join(OUTPUT_DIR, "cvxpy_solution.npy"), z.value)
    print("Saved solution vector to outputs/cvxpy_solution.npy")


if __name__ == "__main__":
    main()