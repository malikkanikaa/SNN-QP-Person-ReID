# SNN-QP: Neurodynamic Spiking Neural Network Quadratic Programming for Person Re-Identification

This repository contains the complete implementation and experimental framework for **Person Re-Identification (Re-ID) Metric Learning** solved via a **Neurodynamic Spiking Neural Network Quadratic Program (SNN-QP)** formulation, evaluated on the **Market-1501** benchmark dataset.

---

## 📌 Project Overview

Traditional metric learning formulations (such as Mahalanobis Distance Learning and Large Margin Nearest Neighbor) can be framed as constrained Quadratic Programming (QP) problems. In this project, we implement and benchmark a bio-inspired **Neurodynamic SNN-QP solver** against standard optimization toolboxes (CVXPY, SciPy SLSQP) and classical metric learning baselines (ITML, LMNN).

### Key Features
1. **Feature Extraction**: Deep feature embeddings extracted using a pretrained `osnet_x1_0` backbone on Market-1501.
2. **Dimension Selection via PCA**: Benchmarking 64-D, 128-D, and 212-D against the full 512-D baseline to identify the optimal performance-efficiency trade-off (selected **128-D**).
3. **Hard-Negative Mining**: Triplet/pair generation sampling positive pairs per identity and mining hardest negatives within candidate pools.
4. **QP Formulation**: Quadratic Program with Frobenius regularization ($\gamma = 0.01$) and slack variable penalties ($\lambda = 1.0$).
5. **Solvers & Benchmarking**:
   - **SNN-QP**: Bio-inspired dynamical system integrating gradient descent and constraint-correction dynamics.
   - **CVXPY**: Convex optimization baseline (OSQP / SCS).
   - **SciPy**: Sequential Least Squares Programming (`SLSQP`).
   - **ITML & LMNN**: Metric learning baselines for distance metric comparison.
6. **Re-ID Evaluation**: Standard Market-1501 evaluation protocol reporting **Rank-1**, **Rank-5**, **Rank-10**, and **mAP**.

---

## 📂 Repository Structure

```text
├── config.py                         # Central pipeline configuration and hyperparameters
├── dataset_utils.py                  # Market-1501 filename parsing and dataset helpers
├── extract_embeddings.py             # Feature extraction using OSNet backbone
├── pca_analysis.py                   # PCA explained variance analysis & curve plotting
├── dimension_eval.py                 # Rank-1 & mAP evaluation across candidate PCA dimensions
├── apply_pca_to_query_gallery.py     # Feature projection to the selected working dimension (128-D)
├── hard_negative_mining.py           # Mining positive and hard-negative pairs
├── qp_cons.py                        # QP matrix construction (Q, c, A, b)
├── run_qp_small_subset.py            # Subset QP builder for solver benchmarking
├── solve_snn.py                      # Neurodynamic SNN-QP solver implementation
├── solve_cvxpy.py                    # CVXPY solver baseline
├── solve_scipy.py                    # SciPy SLSQP solver baseline
├── run_snn_qp_small_subset.py        # SNN-QP execution script on benchmark subset
├── compare_convergence.py            # Convergence and runtime comparison across solvers
├── baseline_itml.py                  # Information-Theoretic Metric Learning (ITML) baseline
├── baseline_lmnn.py                  # Large Margin Nearest Neighbor (LMNN) baseline
├── eval_reid.py                      # Re-ID evaluation pipeline (Rank-k & mAP)
├── reid_eval_utils.py                # Distance metrics and CMC / AP computation utilities
├── render_comparison_table.py        # Markdown/LaTeX comparison table generator
├── requirements.txt                  # Python dependencies
├── models/                           # Model weights storage
├── outputs/                          # Generated embeddings, QP matrices, and plots
└── SNN_QP_final_Report_.docx         # Comprehensive Project Report
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/<YOUR_USERNAME>/<YOUR_REPOSITORY_NAME>.git
cd <YOUR_REPOSITORY_NAME>
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Pipeline Workflow

To reproduce the experiments from scratch, run the scripts in the following order:

### Step 1: Feature Extraction & PCA
```bash
python extract_embeddings.py
python pca_analysis.py
python dimension_eval.py
python apply_pca_to_query_gallery.py
```

### Step 2: Hard-Negative Mining & QP Construction
```bash
python hard_negative_mining.py
python run_qp_small_subset.py
```

### Step 3: Optimization & Solver Comparison
```bash
python solve_cvxpy.py
python solve_scipy.py
python run_snn_qp_small_subset.py
python compare_convergence.py
```

### Step 4: Baseline Benchmark & Final Evaluation
```bash
python baseline_itml.py
python baseline_lmnn.py
python eval_reid.py
python render_comparison_table.py
```

---

## 📊 Results Summary

- **Working Dimension**: Reduced from **512-D** to **128-D** with $< 1\%$ mAP degradation, significantly reducing QP optimization complexity.
- **SNN-QP Solver**: Successfully achieves primal objective convergence and satisfies inequality constraints with comparable precision to interior-point and active-set solvers, while offering potential for neuromorphic hardware acceleration.

---

## 📄 License & Attribution
This project is developed for academic research and evaluation on Person Re-Identification metric learning.
