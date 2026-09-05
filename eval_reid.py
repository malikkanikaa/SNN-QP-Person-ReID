"""
Re-ID Evaluation and Benchmarking Suite for Market-1501.

Evaluates and compares:
  1. Full-D Euclidean Baseline (d = 512, OSNet native features)
  2. PCA-128 Euclidean Baseline (d = 128, unweighted PCA features)
  3. ITML Baseline (Davis et al., 2007, full Mahalanobis M = L^T L)
  4. LMNN Baseline (Weinberger et al., 2009, full Mahalanobis M = L^T L)
  5. SciPy SLSQP (Diagonal QP numerical optimizer baseline, M = diag(w))
  6. CVXPY OSQP (Diagonal QP convex ground truth baseline, M = diag(w))
  7. SNN-QP (Our neurodynamic spiking neural network solver, M = diag(w))

Computes standard Re-ID metrics: Rank-1, Rank-5, Rank-10, Rank-20, and mAP
following official Market-1501 single-query protocol.

Run:
    python eval_reid.py
"""

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import OUTPUT_DIR, PCA_WORKING_DIM
from reid_eval_utils import evaluate_market1501


def evaluate_with_metric(
    query_feats: np.ndarray,
    query_ids: np.ndarray,
    query_cams: np.ndarray,
    gallery_feats: np.ndarray,
    gallery_ids: np.ndarray,
    gallery_cams: np.ndarray,
    L: Optional[np.ndarray] = None,
    ranks: Tuple[int, ...] = (1, 5, 10, 20),
    return_dict: bool = False,
) -> Union[Tuple[float, float, int], Dict[str, Any]]:
    """
    Evaluates Market-1501 Rank-k and mAP after projecting features by linear transformation L.

    Under learned Mahalanobis metric M = L^T L:
        ||L x - L y||_2^2 = (x - y)^T M (x - y)

    Parameters:
        L: (k, d) transformation matrix. If None, uses plain Euclidean distance.
        ranks: Rank cutoffs to compute (default: 1, 5, 10, 20).
        return_dict: If True, returns dict with rank1..rankk, mAP, and cmc array.
    """
    if L is not None:
        L = np.asarray(L)
        query_feats = query_feats @ L.T
        gallery_feats = gallery_feats @ L.T

    return evaluate_market1501(
        query_feats,
        query_ids,
        query_cams,
        gallery_feats,
        gallery_ids,
        gallery_cams,
        ranks=ranks,
        return_dict=return_dict,
    )


def evaluate_with_weights(
    query_feats: np.ndarray,
    query_ids: np.ndarray,
    query_cams: np.ndarray,
    gallery_feats: np.ndarray,
    gallery_ids: np.ndarray,
    gallery_cams: np.ndarray,
    weights: np.ndarray,
    ranks: Tuple[int, ...] = (1, 5, 10, 20),
    return_dict: bool = False,
) -> Union[Tuple[float, float, int], Dict[str, Any]]:
    """
    Evaluates Market-1501 with diagonal metric M = diag(w), w >= 0.
    Equivalent to feature transformation L = diag(sqrt(max(w, 0))).
    """
    w = np.asarray(weights, dtype=float).flatten()
    w_nonneg = np.maximum(w, 0.0)
    scale = np.sqrt(w_nonneg)
    L = np.diag(scale)
    return evaluate_with_metric(
        query_feats,
        query_ids,
        query_cams,
        gallery_feats,
        gallery_ids,
        gallery_cams,
        L=L,
        ranks=ranks,
        return_dict=return_dict,
    )


def load_split_features(
    name: str,
    d: Optional[int] = None,
    output_dir: str = OUTPUT_DIR,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Loads feature embeddings and metadata (identities and camera IDs)."""
    if d is None:
        emb_file = f"{name}_embeddings.npy"
    else:
        emb_file = f"{name}_embeddings_pca{d}.npy"

    emb_path = os.path.join(output_dir, emb_file)
    meta_path = os.path.join(output_dir, f"{name}_meta.csv")

    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"Embeddings file not found: {emb_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    embeddings = np.load(emb_path)
    meta = pd.read_csv(meta_path)
    return embeddings, meta["identity"].values, meta["camera"].values


def render_benchmark_table_image(df: pd.DataFrame, out_path: str):
    """Renders a clean, formatted publication-quality table as PNG."""
    display_df = df.copy()

    # Format percentages
    for col in ["Rank-1 (%)", "Rank-5 (%)", "Rank-10 (%)", "Rank-20 (%)", "mAP (%)"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda v: f"{v:.2f}" if isinstance(v, (int, float)) else str(v))

    fig, ax = plt.subplots(figsize=(10, 0.8 + 0.45 * len(display_df)))
    ax.axis("off")

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1, 1.6)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#1f2937")  # dark slate header
        else:
            method_name = str(display_df.iloc[row - 1]["Method"])
            if "SNN-QP" in method_name:
                cell.set_facecolor("#e0f2fe" if row % 2 == 0 else "#bae6fd")  # highlight our method
                cell.set_text_props(weight="bold" if col == 0 else "normal")
            else:
                cell.set_facecolor("#f9fafb" if row % 2 == 0 else "white")

    plt.title(
        "Market-1501 Re-ID Accuracy Benchmark: SNN-QP vs. Classical Baselines",
        fontsize=12,
        fontweight="bold",
        pad=16,
    )
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    d = PCA_WORKING_DIM
    output_dir = OUTPUT_DIR if os.path.exists(OUTPUT_DIR) else "outputs"
    ranks_to_eval = (1, 5, 10, 20)

    print("=" * 80)
    print("      PERSON Re-ID ACCURACY BENCHMARK (Market-1501 Protocol)")
    print("=" * 80)
    print(f"Working Dimension : d = {d}")
    print(f"Output Directory  : {output_dir}")
    print(f"Evaluated Ranks   : {list(ranks_to_eval)}")
    print()

    # 1. Load PCA-128 features
    print("Loading query and gallery embeddings (PCA-128)...")
    q_emb, q_ids, q_cams = load_split_features("query", d=d, output_dir=output_dir)
    g_emb, g_ids, g_cams = load_split_features("gallery", d=d, output_dir=output_dir)
    print(f"Query split: {q_emb.shape[0]} images | Gallery split: {g_emb.shape[0]} images")

    # Load Full 512-D features if available
    has_full_512 = False
    try:
        q_emb_512, _, _ = load_split_features("query", d=None, output_dir=output_dir)
        g_emb_512, _, _ = load_split_features("gallery", d=None, output_dir=output_dir)
        has_full_512 = True
    except Exception:
        pass

    results_list: List[Dict[str, Any]] = []

    # Helper function to run and record evaluation
    def evaluate_and_record(
        method_name: str,
        category: str,
        L_matrix: Optional[np.ndarray] = None,
        diag_w: Optional[np.ndarray] = None,
        use_512: bool = False,
    ):
        print(f"--> Evaluating: {method_name} ...", end="", flush=True)
        t0 = time.perf_counter()
        if use_512:
            res = evaluate_with_metric(
                q_emb_512, q_ids, q_cams,
                g_emb_512, g_ids, g_cams,
                L=None, ranks=ranks_to_eval, return_dict=True,
            )
        elif diag_w is not None:
            res = evaluate_with_weights(
                q_emb, q_ids, q_cams,
                g_emb, g_ids, g_cams,
                weights=diag_w, ranks=ranks_to_eval, return_dict=True,
            )
        else:
            res = evaluate_with_metric(
                q_emb, q_ids, q_cams,
                g_emb, g_ids, g_cams,
                L=L_matrix, ranks=ranks_to_eval, return_dict=True,
            )
        eval_time = time.perf_counter() - t0
        print(f" Done in {eval_time:.2f}s | Rank-1: {res['rank1']*100:.2f}% | mAP: {res['mAP']*100:.2f}%")

        record = {
            "Category": category,
            "Method": method_name,
            "Rank-1 (%)": round(res["rank1"] * 100, 2),
            "Rank-5 (%)": round(res.get("rank5", 0.0) * 100, 2),
            "Rank-10 (%)": round(res.get("rank10", 0.0) * 100, 2),
            "Rank-20 (%)": round(res.get("rank20", 0.0) * 100, 2),
            "mAP (%)": round(res["mAP"] * 100, 2),
            "Valid Queries": res["num_valid_queries"],
            "Eval Time (s)": round(eval_time, 2),
        }
        results_list.append(record)

    # -----------------------------------------------------------------------
    # 1. Full 512-D Euclidean Baseline (Upper Bound)
    # -----------------------------------------------------------------------
    if has_full_512:
        evaluate_and_record(
            "Full-D (d=512)",
            category="Unweighted Baseline",
            use_512=True,
        )

    # -----------------------------------------------------------------------
    # 2. PCA-128 Euclidean Baseline (Direct unweighted comparator)
    # -----------------------------------------------------------------------
    evaluate_and_record(
        f"Baseline (d={d})",
        category="Unweighted Baseline",
        L_matrix=None,
    )

    # -----------------------------------------------------------------------
    # 3. ITML Baseline (Davis et al., 2007)
    # -----------------------------------------------------------------------
    itml_path = os.path.join(output_dir, "itml_L.npy")
    if os.path.exists(itml_path):
        L_itml = np.load(itml_path)
        evaluate_and_record(
            "ITML",
            category="Metric Learning (Full M)",
            L_matrix=L_itml,
        )
    else:
        print("[-] ITML model file (outputs/itml_L.npy) not found. Run python baseline_itml.py first.")

    # -----------------------------------------------------------------------
    # 4. LMNN Baseline (Weinberger et al., 2009)
    # -----------------------------------------------------------------------
    lmnn_path = os.path.join(output_dir, "lmnn_L.npy")
    if os.path.exists(lmnn_path):
        L_lmnn = np.load(lmnn_path)
        evaluate_and_record(
            "LMNN",
            category="Metric Learning (Full M)",
            L_matrix=L_lmnn,
        )
    else:
        print("[-] LMNN model file (outputs/lmnn_L.npy) not found. Run python baseline_lmnn.py first.")

    # -----------------------------------------------------------------------
    # 5. SciPy SLSQP (Diagonal QP Optimizer)
    # -----------------------------------------------------------------------
    scipy_path = os.path.join(output_dir, "scipy_solution.npy")
    if os.path.exists(scipy_path):
        z_scipy = np.load(scipy_path)
        w_scipy = z_scipy[:d]
        evaluate_and_record(
            "SciPy SLSQP",
            category="QP Optimization (Diag M)",
            diag_w=w_scipy,
        )
    else:
        print("[-] SciPy solution (outputs/scipy_solution.npy) not found. Run python solve_scipy.py first.")

    # -----------------------------------------------------------------------
    # 6. CVXPY OSQP (Convex Ground Truth)
    # -----------------------------------------------------------------------
    cvxpy_path = os.path.join(output_dir, "cvxpy_solution.npy")
    if os.path.exists(cvxpy_path):
        z_cvx = np.load(cvxpy_path)
        w_cvx = z_cvx[:d]
        evaluate_and_record(
            "CVXPY OSQP",
            category="QP Optimization (Diag M)",
            diag_w=w_cvx,
        )
    else:
        print("[-] CVXPY solution (outputs/cvxpy_solution.npy) not found. Run python solve_cvxpy.py first.")

    # -----------------------------------------------------------------------
    # 7. SNN-QP (Our Neurodynamic Spiking Solver)
    # -----------------------------------------------------------------------
    snn_path = os.path.join(output_dir, "snn_qp_solution.npy")
    if os.path.exists(snn_path):
        z_snn = np.load(snn_path)
        w_snn = z_snn[:d]
        evaluate_and_record(
            "SNN-QP",
            category="QP Optimization (Diag M)",
            diag_w=w_snn,
        )
    else:
        print("[-] SNN-QP solution (outputs/snn_qp_solution.npy) not found. Run python solve_snn.py first.")

    # -----------------------------------------------------------------------
    # Present Summary Results Table
    # -----------------------------------------------------------------------
    df_results = pd.DataFrame(results_list)
    print()
    print("=" * 80)
    print("                      FINAL RE-ID BENCHMARK RESULTS")
    print("=" * 80)
    print(df_results.to_string(index=False))
    print("=" * 80)

    # Save to CSV and JSON
    csv_out = os.path.join(output_dir, "reid_benchmark_results.csv")
    json_out = os.path.join(output_dir, "reid_benchmark_results.json")
    df_results.to_csv(csv_out, index=False)
    df_results.to_json(json_out, orient="records", indent=2)
    print(f"\n[+] Saved results to {csv_out}")
    print(f"[+] Saved results to {json_out}")

    # Render PNG table
    png_out = os.path.join(output_dir, "reid_benchmark_table.png")
    render_benchmark_table_image(
        df_results[["Method", "Rank-1 (%)", "Rank-5 (%)", "Rank-10 (%)", "Rank-20 (%)", "mAP (%)"]],
        png_out,
    )
    print(f"[+] Rendered comparison table image to {png_out}")


if __name__ == "__main__":
    main()