"""
Run and verify the new SNN-QP solver on the small feature subset.

Run this AFTER:
    python run_qp_small_subset.py
    python solve_cvxpy.py

Then run:
    python run_snn_qp_small_subset.py

The script:
    1. Loads the saved QP
    2. Extracts the C1 constraints
    3. Runs the new SNN-QP solver
    4. Compares its objective with CVXPY
    5. Saves the solution and convergence traces
"""

import os
import time
import numpy as np

from config import OUTPUT_DIR, SNN_MAX_ITERS, SNN_TOL
from snn_qp_solver_new import SNNQPSolver, split_diagonal_qp


def main():

    # -------------------------------------------------------
    # Load QP
    # -------------------------------------------------------

    Q = np.load(os.path.join(OUTPUT_DIR, "qp_Q.npy"))
    c = np.load(os.path.join(OUTPUT_DIR, "qp_c.npy"))
    A = np.load(os.path.join(OUTPUT_DIR, "qp_A.npy"))
    b = np.load(os.path.join(OUTPUT_DIR, "qp_b.npy"))

    # Infer problem dimensions
    d = int(np.count_nonzero(np.diag(Q)))
    n_neg = Q.shape[0] - d

    print("========================================")
    print("       SNN-QP SMALL SUBSET RUN")
    print("========================================")
    print(f"Q shape       : {Q.shape}")
    print(f"d             : {d}")
    print(f"n_neg         : {n_neg}")
    print(f"Total A rows  : {A.shape[0]}")
    print()

    # -------------------------------------------------------
    # Extract only C1 constraints
    # C2 (xi >= 0) and C3 (w >= 0) are handled internally
    # by the new solver as lower bounds.
    # -------------------------------------------------------

    A_c1, b_c1 = split_diagonal_qp(
        Q, c, A, b, d, n_neg
    )

    print(f"C1 constraint rows used by SNN-QP: {A_c1.shape[0]}")
    print()

    # -------------------------------------------------------
    # Create new SNN-QP solver
    # -------------------------------------------------------

    solver = SNNQPSolver(
        max_iters=SNN_MAX_ITERS,
        k0_scale=0.02,
        constraint_tol=SNN_TOL,
        max_projection_iters=None,
    )

    # -------------------------------------------------------
    # Solve
    # -------------------------------------------------------

    start = time.perf_counter()

    result = solver.solve(
        Q,
        c,
        A_c1,
        b_c1,
        verbose=True
    )

    elapsed = time.perf_counter() - start

    # -------------------------------------------------------
    # Print SNN-QP result
    # -------------------------------------------------------

    print()
    print("========================================")
    print("           SNN-QP RESULT")
    print("========================================")

    print(f"Converged              : "
          f"{result.converged} ({result.convergence_reason})")

    print(f"Iterations             : {result.iterations}")

    print(f"Objective              : "
          f"{result.objective:.6f}")

    print(f"Max constraint viol.   : "
          f"{result.max_constraint_violation:.3e}")

    print(f"k0                     : "
          f"{result.k0:.3e}")

    print(f"Total projections      : "
          f"{result.n_projections}")

    print(f"Max inner iterations   : "
          f"{result.max_inner_iters_seen}")

    print(f"Projection cap hits   : "
          f"{result.projection_budget_exhausted_count}")

    print(f"Gram acceleration     : "
          f"{result.used_gram_acceleration}")

    print(f"SNN-QP solve time      : "
          f"{elapsed:.4f} s")

    # -------------------------------------------------------
    # Compare against CVXPY
    # -------------------------------------------------------

    cvxpy_path = os.path.join(
        OUTPUT_DIR,
        "cvxpy_solution.npy"
    )

    if os.path.exists(cvxpy_path):

        z_cvxpy = np.load(cvxpy_path)

        obj_cvxpy = (
            0.5 * z_cvxpy @ Q @ z_cvxpy
            + c @ z_cvxpy
        )

        gap = result.objective - obj_cvxpy

        rel_gap = gap / (abs(obj_cvxpy) + 1e-12)

        z_dist = (
            np.linalg.norm(result.z - z_cvxpy)
            / (np.linalg.norm(z_cvxpy) + 1e-12)
        )

        print()
        print("========================================")
        print("          CVXPY COMPARISON")
        print("========================================")

        print(f"CVXPY objective        : "
              f"{obj_cvxpy:.6f}")

        print(f"SNN-QP objective       : "
              f"{result.objective:.6f}")

        print(f"Objective gap          : "
              f"{gap:+.6f}")

        print(f"Relative objective gap : "
              f"{rel_gap:+.4%}")

        print(f"Relative solution diff : "
              f"{z_dist:.4%}")

    else:

        print()
        print(
            "cvxpy_solution.npy not found. "
            "Run solve_cvxpy.py first for comparison."
        )

    # -------------------------------------------------------
    # Save results
    # -------------------------------------------------------

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "snn_qp_solution.npy"
        ),
        result.z
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "snn_qp_obj_trace.npy"
        ),
        result.obj_trace
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "snn_qp_viol_trace.npy"
        ),
        result.viol_trace
    )

    print()
    print("========================================")
    print("             FILES SAVED")
    print("========================================")

    print("snn_qp_solution.npy")
    print("snn_qp_obj_trace.npy")
    print("snn_qp_viol_trace.npy")

    print()
    print("Now run:")
    print("    python compare_convergence.py")


if __name__ == "__main__":
    main()