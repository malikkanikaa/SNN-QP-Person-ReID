"""
Plot SNN-QP convergence against CVXPY (OSQP) and SciPy (SLSQP) baselines.
Generates a comprehensive 2x2 comparison plot containing:
  1. Raw Iterates: Objective Trajectory & Constraint Feasibility
  2. Averaged Iterates (Running / Polyak Mean): Objective Trajectory & Constraint Feasibility

Run:
    python compare_convergence.py
"""

import os
import matplotlib

# Non-GUI backend for headless execution
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import OUTPUT_DIR


def compute_running_average(arr: np.ndarray) -> np.ndarray:
    """Computes the cumulative / running average (Polyak-Ruppert / Cesaro mean)."""
    return np.cumsum(arr) / (np.arange(len(arr)) + 1)


def compute_rolling_average(arr: np.ndarray, window: int = 100) -> np.ndarray:
    """Computes rolling moving average."""
    if len(arr) < window:
        return arr
    pad = np.pad(arr, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(pad, np.ones(window) / window, mode="valid")


def main():
    # Resolve output directory
    output_dir = OUTPUT_DIR if os.path.exists(OUTPUT_DIR) else "outputs"

    q_path = os.path.join(output_dir, "qp_Q.npy")
    c_path = os.path.join(output_dir, "qp_c.npy")
    obj_trace_path = os.path.join(output_dir, "snn_qp_obj_trace.npy")
    viol_trace_path = os.path.join(output_dir, "snn_qp_viol_trace.npy")

    if not os.path.exists(obj_trace_path) or not os.path.exists(viol_trace_path):
        raise FileNotFoundError("SNN trace files not found. Run solve_snn.py first.")

    Q = np.load(q_path)
    c = np.load(c_path)
    obj_trace = np.load(obj_trace_path)
    viol_trace = np.load(viol_trace_path)

    # Compute averaged traces
    cum_avg_obj = compute_running_average(obj_trace)
    cum_avg_viol = compute_running_average(viol_trace)

    # Optional rolling average for local trend smoothing
    rolling_window = min(200, max(10, len(obj_trace) // 50))
    rolling_avg_obj = compute_rolling_average(obj_trace, window=rolling_window)

    # Load Baselines
    cvxpy_path = os.path.join(output_dir, "cvxpy_solution.npy")
    scipy_path = os.path.join(output_dir, "scipy_solution.npy")

    obj_cvxpy = None
    if os.path.exists(cvxpy_path):
        z_cvxpy = np.load(cvxpy_path)
        obj_cvxpy = float(0.5 * z_cvxpy @ Q @ z_cvxpy + c @ z_cvxpy)

    obj_scipy = None
    if os.path.exists(scipy_path):
        z_scipy = np.load(scipy_path)
        obj_scipy = float(0.5 * z_scipy @ Q @ z_scipy + c @ z_scipy)

    # -----------------------------------------------------------------------
    # Create 2x2 Grid Figure (Raw vs. Averaged)
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)

    # -----------------------------------------------------------------------
    # Row 1, Col 1: RAW Objective Trajectory
    # -----------------------------------------------------------------------
    ax = axes[0, 0]
    ax.plot(obj_trace, label="SNN-QP (Raw)", color="#1f77b4", alpha=0.9, lw=1.2)
    if obj_cvxpy is not None:
        ax.axhline(obj_cvxpy, color="#2ca02c", linestyle="--", lw=1.8,
                   label=f"CVXPY Ground Truth ({obj_cvxpy:.4f})")
    if obj_scipy is not None:
        ax.axhline(obj_scipy, color="#ff7f0e", linestyle=":", lw=1.8,
                   label=f"SciPy SLSQP ({obj_scipy:.4f})")

    ax.set_ylabel("Objective Value", fontsize=11, fontweight="bold")
    ax.set_title("Raw Iterates: Objective Convergence", fontsize=12, fontweight="bold", pad=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9, loc="upper right")

    # -----------------------------------------------------------------------
    # Row 1, Col 2: RAW Constraint Violation
    # -----------------------------------------------------------------------
    ax = axes[0, 1]
    # Clamp to small positive value to avoid log(0)
    safe_viol = np.maximum(viol_trace, 1e-12)
    ax.plot(safe_viol, color="#d62728", lw=1.2, label="Max Constraint Viol. (Raw)")
    ax.set_yscale("log")
    ax.set_ylabel("Max Violation (Log Scale)", fontsize=11, fontweight="bold")
    ax.set_title("Raw Iterates: Constraint Feasibility", fontsize=12, fontweight="bold", pad=8)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(fontsize=9, loc="upper right")

    # -----------------------------------------------------------------------
    # Row 2, Col 1: AVERAGED Objective Trajectory
    # -----------------------------------------------------------------------
    ax = axes[1, 0]
    ax.plot(obj_trace, color="#1f77b4", alpha=0.2, lw=0.8, label="SNN-QP (Raw Iterates)")
    ax.plot(cum_avg_obj, label="SNN-QP (Running Cumulative Mean)", color="#9467bd", lw=2.0)
    ax.plot(rolling_avg_obj, label=f"SNN-QP (Rolling Mean, w={rolling_window})", color="#17becf", linestyle="-.", lw=1.6)

    if obj_cvxpy is not None:
        ax.axhline(obj_cvxpy, color="#2ca02c", linestyle="--", lw=1.8,
                   label=f"CVXPY Ground Truth ({obj_cvxpy:.4f})")
    if obj_scipy is not None:
        ax.axhline(obj_scipy, color="#ff7f0e", linestyle=":", lw=1.8,
                   label=f"SciPy SLSQP ({obj_scipy:.4f})")

    ax.set_xlabel("SNN-QP Iteration", fontsize=11, fontweight="bold")
    ax.set_ylabel("Objective Value", fontsize=11, fontweight="bold")
    ax.set_title("Averaged Iterates: Objective Convergence", fontsize=12, fontweight="bold", pad=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9, loc="upper right")

    # -----------------------------------------------------------------------
    # Row 2, Col 2: AVERAGED Constraint Violation
    # -----------------------------------------------------------------------
    ax = axes[1, 1]
    safe_cum_viol = np.maximum(cum_avg_viol, 1e-12)
    ax.plot(safe_viol, color="#d62728", alpha=0.25, lw=0.8, label="Raw Violations")
    ax.plot(safe_cum_viol, color="#e377c2", lw=2.0, label="Running Cumulative Mean Viol.")
    ax.set_yscale("log")
    ax.set_xlabel("SNN-QP Iteration", fontsize=11, fontweight="bold")
    ax.set_ylabel("Max Violation (Log Scale)", fontsize=11, fontweight="bold")
    ax.set_title("Averaged Iterates: Constraint Feasibility", fontsize=12, fontweight="bold", pad=8)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(fontsize=9, loc="upper right")

    fig.suptitle("SNN-QP Convergence Analysis: Raw vs. Averaged Dynamics", fontsize=14, fontweight="bold", y=0.99)
    fig.tight_layout()

    out_unified = os.path.join(output_dir, "snn_qp_convergence.png")
    fig.savefig(out_unified, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved unified 2x2 convergence plot (Raw + Averaged) to: {out_unified}")

    # -----------------------------------------------------------------------
    # Also save separate stand-alone figures for quick report inclusion
    # -----------------------------------------------------------------------
    # 1. Stand-alone Raw Figure
    fig_raw, axes_raw = plt.subplots(1, 2, figsize=(12, 4.5))
    axes_raw[0].plot(obj_trace, label="SNN-QP", color="#1f77b4", lw=1.5)
    if obj_cvxpy is not None:
        axes_raw[0].axhline(obj_cvxpy, color="#2ca02c", linestyle="--", label=f"CVXPY ({obj_cvxpy:.4f})")
    if obj_scipy is not None:
        axes_raw[0].axhline(obj_scipy, color="#ff7f0e", linestyle=":", label=f"SciPy ({obj_scipy:.4f})")
    axes_raw[0].set_xlabel("Iteration")
    axes_raw[0].set_ylabel("Objective")
    axes_raw[0].set_title("Raw Objective Convergence")
    axes_raw[0].legend()
    axes_raw[0].grid(True, linestyle="--", alpha=0.5)

    axes_raw[1].plot(safe_viol, color="#d62728", lw=1.5)
    axes_raw[1].set_yscale("log")
    axes_raw[1].set_xlabel("Iteration")
    axes_raw[1].set_ylabel("Max Violation (Log Scale)")
    axes_raw[1].set_title("Raw Feasibility")
    axes_raw[1].grid(True, which="both", linestyle="--", alpha=0.5)
    fig_raw.tight_layout()
    out_raw = os.path.join(output_dir, "snn_qp_convergence_raw.png")
    fig_raw.savefig(out_raw, dpi=150, bbox_inches="tight")
    plt.close(fig_raw)
    print(f"Saved stand-alone raw plot to: {out_raw}")

    # 2. Stand-alone Averaged Figure
    fig_avg, axes_avg = plt.subplots(1, 2, figsize=(12, 4.5))
    axes_avg[0].plot(obj_trace, color="#1f77b4", alpha=0.2, label="Raw")
    axes_avg[0].plot(cum_avg_obj, label="Cumulative Average", color="#9467bd", lw=2.0)
    if obj_cvxpy is not None:
        axes_avg[0].axhline(obj_cvxpy, color="#2ca02c", linestyle="--", label=f"CVXPY ({obj_cvxpy:.4f})")
    axes_avg[0].set_xlabel("Iteration")
    axes_avg[0].set_ylabel("Objective")
    axes_avg[0].set_title("Averaged Objective Convergence")
    axes_avg[0].legend()
    axes_avg[0].grid(True, linestyle="--", alpha=0.5)

    axes_avg[1].plot(safe_cum_viol, color="#e377c2", lw=2.0, label="Cumulative Avg Viol.")
    axes_avg[1].set_yscale("log")
    axes_avg[1].set_xlabel("Iteration")
    axes_avg[1].set_ylabel("Max Violation (Log Scale)")
    axes_avg[1].set_title("Averaged Feasibility")
    axes_avg[1].grid(True, which="both", linestyle="--", alpha=0.5)
    fig_avg.tight_layout()
    out_avg = os.path.join(output_dir, "snn_qp_convergence_averaged.png")
    fig_avg.savefig(out_avg, dpi=150, bbox_inches="tight")
    plt.close(fig_avg)
    print(f"Saved stand-alone averaged plot to: {out_avg}")


if __name__ == "__main__":
    main()