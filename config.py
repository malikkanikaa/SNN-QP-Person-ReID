"""
Central config for the SNN-QP embedding extraction + PCA pipeline.
Edit the paths below to match your machine, then leave the other
scripts untouched.
"""

import os

# ---- Dataset location ------------------------------------------------------
DATASET_ROOT = r"D:\archive\Market-1501-v15.09.15"

TRAIN_DIR = os.path.join(DATASET_ROOT, "bounding_box_train")
TEST_DIR = os.path.join(DATASET_ROOT, "bounding_box_test")   # gallery
QUERY_DIR = os.path.join(DATASET_ROOT, "query")

# ---- Where results get saved -----------------------------------------------
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Backbone model ---------------------------------------------------------
# osnet_x1_0 outputs 512-D features natively.
# For the performance-based dimension evaluation, use the same
# pretrained Market-1501 weights used to extract the embeddings.

MODEL_NAME = "osnet_x1_0"
MODEL_WEIGHTS_PATH = r"D:\FTP\models\osnet_x1_market.pth"

EMBED_DIM = 512
BATCH_SIZE = 64
DEVICE = "cuda"  # extract_embeddings.py falls back to "cpu" automatically

# ---- PCA settings -----------------------------------------------------------
# PCA is fitted on TRAIN embeddings only and then applied to
# train/query/gallery embeddings.
#
# These settings are kept for reference/analysis. Dimension selection
# is now based on Rank-1 and mAP performance rather than explained variance.

PCA_CANDIDATE_DIMS = [64, 128, 212]
VARIANCE_THRESHOLD = 0.95

# ---- Rank-1 / mAP dimension comparison -------------------------------------
# Candidate reduced dimensions to benchmark against the full 512-D embedding.
EVAL_DIMS = [64, 128, 212]

# Full embedding dimension used as the performance baseline.
BASELINE_DIM = 512

# A candidate dimension is considered close enough to the full-dimensional
# baseline if its mAP is within this absolute difference.
# Example: 0.01 = 1 percentage point.
MAP_TOLERANCE = 0.01

 
# ---- Chosen working dimension (Section 3 decision) -------------------------
PCA_WORKING_DIM = 128
 
# ---- Hard-negative mining (Week 1-2, remaining step) -----------------------
POS_PAIRS_PER_ID = 3     # positive pairs sampled per identity
NEG_POOL_SIZE = 300      # random candidate pool size per anchor before mining
HARD_NEG_K = 5           # hardest negatives kept per anchor - keeps |N| small
RANDOM_SEED = 42
 
# ---- QP construction (Week 3-4) --------------------------------------------
GAMMA_REG = 0.01         # Frobenius regularisation coefficient (gamma)
LAMBDA_REG = 1.0         # slack penalty coefficient (lambda)
MARGIN = 1.0             # margin m
SMALL_SUBSET_SIZE = 200  # pairs used for CVXPY/SciPy profiling (Week 3-4)
 
# ---- SNN-QP neurodynamic solver (Week 5-6) ---------------------------------
#SNN_ETA = 1e-3           # primal (gradient-descent block) step size
#SNN_ETA_C = 1e-2         # constraint-correction block step size / gain
SNN_MAX_ITERS = 30000     # Euler integration steps ("spike rounds")
SNN_TOL = 1e-10           # early-stop tolerance on relative change in z
