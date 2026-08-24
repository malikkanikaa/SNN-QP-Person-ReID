"""
Helpers for parsing Market-1501 filenames into (identity, camera) labels
and building a file list per split.

Market-1501 filename format:  0001_c1s1_001051_00.jpg
    0001   -> identity (person ID). '-1' and '0000' are junk/distractor
              images and are excluded from training pairs.
    c1     -> camera ID (1-6)
    s1     -> sequence (not needed here)
    001051 -> frame number
    00     -> box index within frame
"""

import os
import re
import pandas as pd

FILENAME_PATTERN = re.compile(r"(-?\d+)_c(\d)s(\d)_(\d+)_(\d+)")


def parse_filename(filename):
    m = FILENAME_PATTERN.match(filename)
    if not m:
        return None
    identity = int(m.group(1))
    camera = int(m.group(2))
    return identity, camera


def build_file_list(folder, exclude_junk=True):
    """
    Returns a DataFrame with columns: path, identity, camera
    for every .jpg in `folder`.

    exclude_junk: drop identity -1 (junk) and 0000 (background) images.
    Set True for the training split, False for query/gallery (evaluation
    protocol needs the junk images present, even if scored specially).
    """
    records = []
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith(".jpg"):
            continue
        parsed = parse_filename(fname)
        if parsed is None:
            continue
        identity, camera = parsed
        if exclude_junk and identity in (-1, 0):
            continue
        records.append({
            "path": os.path.join(folder, fname),
            "identity": identity,
            "camera": camera,
        })
    return pd.DataFrame(records)


if __name__ == "__main__":
    # Quick sanity check - run this file directly to confirm paths are right
    # before running the heavier extraction script.
    from config import TRAIN_DIR, QUERY_DIR, TEST_DIR

    for name, folder in [("train", TRAIN_DIR), ("query", QUERY_DIR), ("gallery", TEST_DIR)]:
        df = build_file_list(folder, exclude_junk=(name == "train"))
        print(f"{name}: {len(df)} images, {df['identity'].nunique()} identities")