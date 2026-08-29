"""
Generate a summary figure for cross-dataset transfer evaluation.

This script reads pairwise transfer and LODO transfer results and generates
a figure with:
- Left: Heatmap of pairwise transfer results
- Right: Bar chart of LODO results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from _common import RESULTS_DIR, FIGURES_DIR

DATASET_KEYS = ("torres_torres", "ibarra_zarate", "raeisi", "wang")


def main():
    # Read both CSVs
    pairwise_df = pd.read_csv(RESULTS_DIR / "pairwise_transfer.csv")
    lodo_df = pd.read_csv(RESULTS_DIR / "lodo_transfer.csv")
    
    # Build 4x4 pairwise matrix for balanced_accuracy
    # rows = source dataset, cols = target dataset
    matrix = np.full((4, 4), np.nan)
    
    # Map dataset keys to indices
    key_to_idx = {key: i for i, key in enumerate(DATASET_KEYS)}
    
    # Fill the matrix
    for _, row in pairwise_df.iterrows():
        source_idx = key_to_idx[row["source"]]
        target_idx = key_to_idx[row["target"]]
        matrix[source_idx, target_idx] = row["balanced_accuracy"]
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # LEFT: Pairwise heatmap
    im = axes[0].imshow(matrix, cmap="viridis", vmin=0, vmax=1, aspect="equal")
    
    # Set ticks and labels
    axes[0].set_xticks(range(4))
    axes[0].set_xticklabels(DATASET_KEYS, rotation=45, ha="right")
    axes[0].set_yticks(range(4))
    axes[0].set_yticklabels(DATASET_KEYS)
    
    # Add colorbar
    fig.colorbar(im, ax=axes[0])
    
    # Title
    axes[0].set_title("Pairwise transfer: balanced accuracy (source -> target)")
    
    # Annotate cells with values
    for i in range(4):
        for j in range(4):
            if i != j:  # Skip diagonal
                axes[0].text(j, i, f"{matrix[i, j]:.2f}", 
                           ha="center", va="center", color="white")
    
    # RIGHT: LODO bar chart
    lodo_values = lodo_df.set_index("held_out").loc[list(DATASET_KEYS), "balanced_accuracy"].values
    axes[1].bar(DATASET_KEYS, lodo_values)
    
    # Set labels and title
    axes[1].set_ylabel("balanced accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("LODO: held-out dataset balanced accuracy")
    
    # Rotate x tick labels
    plt.xticks(rotation=30)
    
    # Overall title
    fig.suptitle("Cross-dataset transfer (Tier 1 elastic-net)")
    fig.tight_layout()
    
    # Save figure
    path = FIGURES_DIR / "tier1" / "LODO" / "cross_dataset_transfer_summary.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    print(f"Saved figure to {path}")


if __name__ == "__main__":
    main()