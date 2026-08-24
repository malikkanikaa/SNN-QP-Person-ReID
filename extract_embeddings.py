"""
Extract 512-D OSNet embeddings for the Market-1501 train / query / gallery
splits and save them to disk as .npy (+ a metadata .csv with identity/camera
labels, needed later for hard-negative mining).

Run: python extract_embeddings.py
"""

import os
import numpy as np
import torch
from torchreid.reid.utils import FeatureExtractor

from config import (
    TRAIN_DIR, QUERY_DIR, TEST_DIR, OUTPUT_DIR,
    MODEL_NAME, MODEL_WEIGHTS_PATH, BATCH_SIZE, DEVICE,
)
from dataset_utils import build_file_list


def get_device():
    return DEVICE if torch.cuda.is_available() else "cpu"


def extract_split(extractor, df, split_name):
    emb_path = os.path.join(
        OUTPUT_DIR, f"{split_name}_embeddings.npy"
    )
    meta_path = os.path.join(
        OUTPUT_DIR, f"{split_name}_meta.csv"
    )

    # If this split is already complete, skip it
    if os.path.exists(emb_path) and os.path.exists(meta_path):
        existing_embeddings = np.load(emb_path)

        if len(existing_embeddings) == len(df):
            print(
                f"[{split_name}] Already complete "
                f"({existing_embeddings.shape}). Skipping."
            )
            return existing_embeddings

        else:
            print(
                f"[{split_name}] Existing file has wrong number "
                f"of embeddings. Recomputing."
            )

    paths = df["path"].tolist()
    embeddings = []

    for start in range(0, len(paths), BATCH_SIZE):
        batch_paths = paths[start:start + BATCH_SIZE]

        feats = extractor(batch_paths)

        embeddings.append(feats.cpu().numpy())

        print(
            f"[{split_name}] "
            f"{start + len(batch_paths)}/{len(paths)}",
            end="\r"
        )

    embeddings = np.concatenate(embeddings, axis=0)

    print(
        f"\n[{split_name}] "
        f"embeddings shape: {embeddings.shape}"
    )

    np.save(emb_path, embeddings)
    df.to_csv(meta_path, index=False)

    return embeddings

def main():
    device = get_device()
    print(f"Using device: {device}")

    extractor = FeatureExtractor(
        model_name=MODEL_NAME,
        model_path=MODEL_WEIGHTS_PATH,   # None -> ImageNet-init backbone only
        device=device,
    )

    train_df = build_file_list(TRAIN_DIR, exclude_junk=True)
    query_df = build_file_list(QUERY_DIR, exclude_junk=False)
    gallery_df = build_file_list(TEST_DIR, exclude_junk=False)

    print(f"train: {len(train_df)} | query: {len(query_df)} | gallery: {len(gallery_df)}")

    extract_split(extractor, train_df, "train")
    extract_split(extractor, query_df, "query")
    extract_split(extractor, gallery_df, "gallery")

    print("\nDone. Files saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()