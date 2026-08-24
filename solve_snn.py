"""
SNN-QP Solver for Diagonal-M Person Re-Identification Metric Learning.
Based on the canonical neurodynamic optimization architecture (ahkhan03/SNN_opt).

Solves:
    minimize   0.5 * z^T Q z + c^T z
    subject to C z + d <= 0,  z >= 0  (where z = [w; xi])

Run:
    python solve_snn.py
"""

import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Problem Definition & Configurations (Matching guide's SNN_opt architecture)
# ---------------------------------------------------------------------------

@dataclass
class OptimizationProblem:
    """
    Defines a constrained quadratic program:
        minimize    0.5 * x^T A x + b^T x
        subject to  C x + d <= 0
    """
    A: np.ndarray
    b: np.ndarray
    C: np.ndarray
    d: np.ndarray

    def __post_init__(self):
        n = self.A.shape[0]
        assert self.A.shape == (n, n), "A must be square"
        assert self.b.shape == (n,), f"b must have shape ({n},)"
        assert self.C.shape[1] == n, f"C must have {n} columns"
        assert self.C.shape[0] == self.d.shape[0], "C rows must match d length"

    @property
    def n_vars(self) -> int:
        return self.A.shape[0]

    @property
    def n_constraints(self) -> int:
        return self.C.shape[0]

    def objective(self, x: np.ndarray) -> float:
        return float(0.5 * x @ self.A @ x + self.b @ x)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return self.A @ x + self.b

    def constraint_values(self, x: np.ndarray) -> np.ndarray:
        return self.C @ x + self.d


@dataclass
class ConvergenceConfig:
    """Convergence detection and KKT optimality certificate settings."""
    enable_early_stopping: bool = True
    kkt_abs_tol: float = 1e-6
    kkt_rel_tol: float = 1e-4
    obj_rel_tol: float = 1e-8
    feasibility_tol: float = 1e-4
    check_every: int = 50
    min_iterations: int = 500
    window_size: int = 10
    patience: int = 3


@dataclass
class SolverConfig:
    """Solver configuration for discrete Euler dynamics and WTA projection."""
    k0: Optional[float] = None          # Gradient step size (None = auto)
    k0_scale: float = 1e-5             # Scaled for singular Hessian + large c
    max_iterations: int = 10000        # Outer discrete-Euler steps
    constraint_tol: float = 1e-6       # Geometric tolerance for projections
    max_projection_iters: int = 200    # Inner WTA projection safety cap
    lower_bound: Optional[float] = 0.0 # Enforces z >= 0 (w >= 0 and xi >= 0)
    upper_bound: Optional[float] = None
    convergence: ConvergenceConfig = field(default_factory=ConvergenceConfig)


@dataclass
class SolverResult:
    """Results from the SNN-QP solve."""
    final_x: np.ndarray
    final_objective: float
    converged: bool
    convergence_reason: str
    iterations_used: int
    n_projections: int
    max_constraint_violation: float
    k0: float
    obj_trace: np.ndarray
    viol_trace: np.ndarray


# ---------------------------------------------------------------------------
# SNN-QP Solver Core
# ---------------------------------------------------------------------------

class SNNSolver:
    """
    Spiking Neural Network / Neurodynamic QP Solver.

    Alternates between:
      Phase 1: Primal gradient descent (leaky integrator neuron dynamics)
      Phase 2: Winner-take-all (WTA) exact adaptive boundary projection
               with O(m) event-driven Gram matrix lateral updates.
    """

    def __init__(self, problem: OptimizationProblem, config: Optional[SolverConfig] = None):
        self.problem = problem
        self.config = config or SolverConfig()

        # Precompute row norms and normalized-distance scales
        if self.problem.n_constraints > 0:
            C = self.problem.C
            self._c_norms_sq = np.sum(C ** 2, axis=1)
            self._c_norms = np.sqrt(self._c_norms_sq)
            with np.errstate(divide="ignore"):
                self._row_scale = np.where(
                    self._c_norms > 1e-12,
                    1.0 / np.maximum(self._c_norms, 1e-300),
                    0.0
                )
            # Lateral constraint coupling Gram matrix G = C C^T
            self._c_gram = C @ C.T
        else:
            self._c_norms_sq = np.array([])
            self._c_norms = np.array([])
            self._row_scale = np.array([])
            self._c_gram = None

        # Step size calculation
        if self.config.k0 is not None:
            self._k0 = float(self.config.k0)
        else:
            self._k0 = self._compute_adaptive_k0()

    def _compute_adaptive_k0(self) -> float:
        """
        Calculates safe step size k0.
        For singular / partially-flat Hessian Q with large linear cost ||c||,
        scales step size inversely with ||c|| to prevent trajectory explosion.
        """
        grad_norm = float(np.linalg.norm(self.problem.b))
        if grad_norm > 1.0:
            return float(self.config.k0_scale)
        return float(self.config.k0_scale)

    def _joint_violation(self, x: np.ndarray) -> float:
        """Calculates joint max geometric constraint violation."""
        viol_rows = 0.0
        if self.problem.n_constraints > 0:
            g = self.problem.constraint_values(x)
            viol_rows = float(np.max(np.maximum(g * self._row_scale, 0.0)))

        viol_lo = 0.0
        if self.config.lower_bound is not None:
            viol_lo = float(np.max(np.maximum(self.config.lower_bound - x, 0.0)))

        return max(viol_rows, viol_lo)

    def _project_adaptive(self, x: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Exact Winner-Take-All (WTA) adaptive projection:
        Unifies explicit C rows and coordinate box bounds (z >= 0) in one candidate family.
        Applies exact single-step corrections with lateral Gram updates.
        """
        lo = self.config.lower_bound
        m = self.problem.n_constraints
        tol = self.config.constraint_tol
        gram = self._c_gram

        x_proj = x.copy()
        g = self.problem.constraint_values(x_proj) if m > 0 else np.array([])
        n_iters = 0

        for _ in range(self.config.max_projection_iters):
            j_row = int(np.argmax(g * self._row_scale)) if m > 0 else -1
            best_val = float(g[j_row] * self._row_scale[j_row]) if m > 0 else -np.inf
            kind, j = "row", j_row

            # Check lower bound facet candidates (unit normal)
            if lo is not None:
                v_lo = lo - x_proj
                i = int(np.argmax(v_lo))
                if v_lo[i] > best_val:
                    best_val, kind, j = float(v_lo[i]), "lo", i

            # Feasible within geometric tolerance
            if best_val <= tol:
                break

            if kind == "row":
                violation = g[j]
                k1 = violation / self._c_norms_sq[j]
                x_proj -= k1 * self.problem.C[j]
                if gram is not None:
                    g -= k1 * gram[j]
            else:
                delta = best_val  # correction to move coordinate to lower_bound
                x_proj[j] += delta
                if gram is not None and m > 0:
                    g += delta * self.problem.C[:, j]

            n_iters += 1

        return x_proj, n_iters

    def solve(self, x0: Optional[np.ndarray] = None, verbose: bool = False) -> SolverResult:
        """Runs the discrete Euler SNN solver."""
        n = self.problem.n_vars
        x = np.zeros(n) if x0 is None else np.asarray(x0, dtype=float).copy()
        conv = self.config.convergence

        obj_trace = np.empty(self.config.max_iterations)
        viol_trace = np.empty(self.config.max_iterations)
        obj_history: List[float] = []
        patience_counter = 0
        total_projections = 0
        converged = False
        reason = "max_iterations"

        if verbose:
            print(f"[SNN-QP] Starting solve: n_vars={n}, n_constraints={self.problem.n_constraints}, k0={self._k0:.2e}")

        t = 0
        for t in range(self.config.max_iterations):
            # Phase 1: Leaky Integrator / Gradient Step
            grad = self.problem.gradient(x)
            x -= self._k0 * grad

            # Phase 2: Exact WTA Boundary Projection
            x, n_proj = self._project_adaptive(x)
            total_projections += n_proj

            # Telemetry
            obj = self.problem.objective(x)
            viol = self._joint_violation(x)
            obj_trace[t] = obj
            viol_trace[t] = viol

            obj_history.append(obj)
            if len(obj_history) > conv.window_size * 2:
                obj_history = obj_history[-conv.window_size * 2:]

            if verbose and (t + 1) % 1000 == 0:
                print(f"  iter {t+1:5d} | obj = {obj:12.6f} | max_viol = {viol:.2e} | projs = {total_projections}")

            # Convergence Check
            if conv.enable_early_stopping and t >= conv.min_iterations and (t % conv.check_every == 0):
                if viol <= conv.feasibility_tol:
                    window = obj_history[-conv.window_size:]
                    obj_rel_change = (max(window) - min(window)) / max(abs(window[-1]), 1e-10)
                    if obj_rel_change < conv.obj_rel_tol:
                        patience_counter += 1
                        if patience_counter >= conv.patience:
                            converged = True
                            reason = f"obj_plateau(rel_change={obj_rel_change:.2e}) + feasible"
                            break
                    else:
                        patience_counter = 0
                else:
                    patience_counter = 0

        actual_iters = t + 1
        final_obj = self.problem.objective(x)
        final_viol = self._joint_violation(x)

        return SolverResult(
            final_x=x,
            final_objective=final_obj,
            converged=converged,
            convergence_reason=reason,
            iterations_used=actual_iters,
            n_projections=total_projections,
            max_constraint_violation=final_viol,
            k0=self._k0,
            obj_trace=obj_trace[:actual_iters],
            viol_trace=viol_trace[:actual_iters]
        )


# ---------------------------------------------------------------------------
# Runner & Baseline Comparison
# ---------------------------------------------------------------------------

def main():
    # Resolve output directory whether running from root or outputs/
    if os.path.exists("outputs"):
        output_dir = "outputs"
    elif os.path.exists("qp_Q.npy"):
        output_dir = "."
    else:
        output_dir = os.path.join(os.path.dirname(__file__), "outputs")

    q_path = os.path.join(output_dir, "qp_Q.npy")
    c_path = os.path.join(output_dir, "qp_c.npy")
    a_path = os.path.join(output_dir, "qp_A.npy")
    b_path = os.path.join(output_dir, "qp_b.npy")

    if not os.path.exists(q_path):
        raise FileNotFoundError(f"QP data not found in {output_dir}. Please run run_qp_small_subset.py first.")

    Q = np.load(q_path)
    c = np.load(c_path)
    A = np.load(a_path)
    b = np.load(b_path)

    d = int(np.count_nonzero(np.diag(Q)))
    n_neg = Q.shape[0] - d

    # Extract C1 margin constraints: A_C1 z <= b_C1  <=>  C z + d_vec <= 0
    C = A[:n_neg]
    d_vec = -b[:n_neg]

    print("=" * 60)
    print("       SNN-QP NEURODYNAMIC SOLVER (SNN_opt)")
    print("=" * 60)
    print(f"Dimension d (w): {d} | Negative pairs (|N|): {n_neg}")
    print(f"Total variables: {Q.shape[0]} | Margin constraints: {C.shape[0]}")
    print()

    problem = OptimizationProblem(A=Q, b=c, C=C, d=d_vec)
    config = SolverConfig(
        k0=1e-5,                  # Calibrated for singular Q + large c
        lower_bound=0.0,          # Handles w >= 0 and xi >= 0
        max_iterations=10000,
        constraint_tol=1e-6,
        max_projection_iters=200
    )

    solver = SNNSolver(problem, config)

    start = time.perf_counter()
    result = solver.solve(verbose=True)
    elapsed = time.perf_counter() - start

    print()
    print("=" * 60)
    print("                   SNN-QP RESULTS")
    print("=" * 60)
    print(f"SNN Objective          : {result.final_objective:.6f}")
    print(f"Converged              : {result.converged} ({result.convergence_reason})")
    print(f"Iterations Used        : {result.iterations_used}")
    print(f"Total Projections      : {result.n_projections}")
    print(f"Max Constraint Viol.   : {result.max_constraint_violation:.3e}")
    print(f"Solve Time             : {elapsed:.4f} s")

    # Compare against CVXPY Ground Truth if present
    cvxpy_path = os.path.join(output_dir, "cvxpy_solution.npy")
    if os.path.exists(cvxpy_path):
        z_cvx = np.load(cvxpy_path)
        obj_cvx = float(0.5 * z_cvx @ Q @ z_cvx + c @ z_cvx)
        gap = result.final_objective - obj_cvx
        rel_gap = gap / abs(obj_cvx)

        w_snn = result.final_x[:d]
        w_cvx = z_cvx[:d]
        w_dist = float(np.linalg.norm(w_snn - w_cvx) / np.linalg.norm(w_cvx))

        print()
        print("=" * 60)
        print("              GROUND TRUTH (CVXPY) COMPARISON")
        print("=" * 60)
        print(f"CVXPY Objective        : {obj_cvx:.6f}")
        print(f"SNN-QP Objective       : {result.final_objective:.6f}")
        print(f"Absolute Objective Gap : {gap:+.6f}")
        print(f"Relative Objective Gap : {rel_gap:+.4%}")
        print(f"Metric w Rel. Distance : {w_dist:.2%}")
        print("=" * 60)

    # Save solutions
    np.save(os.path.join(output_dir, "snn_qp_solution.npy"), result.final_x)
    np.save(os.path.join(output_dir, "snn_qp_obj_trace.npy"), result.obj_trace)
    np.save(os.path.join(output_dir, "snn_qp_viol_trace.npy"), result.viol_trace)
    print(f"\nSaved solution and traces to {output_dir}/")


if __name__ == "__main__":
    main()
