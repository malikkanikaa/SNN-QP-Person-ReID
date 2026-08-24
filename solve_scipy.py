"""
Solve the same diagonal-M QP with SciPy's SLSQP - the interior-point /
active-set reference trajectory referenced in report Section 4.2.

Compare the printed objective value against solve_cvxpy.py's output to
verify solver correctness (Section 4.3).

Run this AFTER run_qp_small_subset.py.
Run: python solve_scipy.py
"""

import os
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint

from config import OUTPUT_DIR


def main():
    Q = np.load(os.path.join(OUTPUT_DIR, "qp_Q.npy"))
    c = np.load(os.path.join(OUTPUT_DIR, "qp_c.npy"))
    A = np.load(os.path.join(OUTPUT_DIR, "qp_A.npy"))
    b = np.load(os.path.join(OUTPUT_DIR, "qp_b.npy"))

    n_vars = Q.shape[0]

    def objective(z):
        return 0.5 * z @ Q @ z + c @ z

    def objective_grad(z):
        return Q @ z + c

    # A z <= b  ->  SciPy's LinearConstraint form: lb <= A z <= ub
    linear_constraint = LinearConstraint(A, -np.inf, b)

    z0 = np.zeros(n_vars)

    start = time.perf_counter()
    result = minimize(
        objective, z0, jac=objective_grad, method="SLSQP",
        constraints=[linear_constraint],
        options={"maxiter": 500, "ftol": 1e-9},
    )
    elapsed = time.perf_counter() - start

    print(f"SciPy SLSQP success: {result.success}")
    print(f"Message: {result.message}")
    print(f"Objective value: {result.fun:.6f}")
    print(f"Iterations: {result.nit}")
    print(f"Solve time: {elapsed:.4f} s")

    np.save(os.path.join(OUTPUT_DIR, "scipy_solution.npy"), result.x)
    print("Saved solution vector to outputs/scipy_solution.npy")


if __name__ == "__main__":
    main()