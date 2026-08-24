"""
Render dimension_eval_results.csv as a formatted PNG table, for direct
inclusion in the report as the dimension-choice justification figure.

Run this AFTER dimension_eval.py.
Run: python render_comparison_table.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

from config import OUTPUT_DIR


def main():
    results_csv = os.path.join(OUTPUT_DIR, "dimension_eval_results.csv")
    df = pd.read_csv(results_csv).sort_values("dimension").reset_index(drop=True)

    display_df = pd.DataFrame({
        "Dimension (d)": df["dimension"].apply(lambda d: "512 (full)" if d == 512 else str(int(d))),
        "Rank-1 (%)": (df["rank1"] * 100).round(2),
        "mAP (%)": (df["mAP"] * 100).round(2),
    })

    fig, ax = plt.subplots(figsize=(6, 0.6 + 0.5 * len(display_df)))
    ax.axis("off")

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#2c3e50")
        else:
            cell.set_facecolor("#f5f5f5" if row % 2 == 0 else "white")

    plt.title("Rank-1 / mAP vs. Embedding Dimension (Market-1501)", fontsize=12, pad=14)

    out_path = os.path.join(OUTPUT_DIR, "dimension_comparison_table.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved comparison table image to {out_path}")


if __name__ == "__main__":
    main()