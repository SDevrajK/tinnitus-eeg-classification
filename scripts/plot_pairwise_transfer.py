"""
Pairwise cross-dataset transfer matrix figure for Tiers 1–3.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from _common import RESULTS_DIR, FIGURES_DIR

def main():
    # Load the three CSV files
    df_tier1 = pd.read_csv(RESULTS_DIR / "pairwise_transfer_tier1.csv")
    df_tier2 = pd.read_csv(RESULTS_DIR / "pairwise_transfer_tier2.csv")
    df_tier3 = pd.read_csv(RESULTS_DIR / "pairwise_transfer_tier3.csv")
    df = pd.concat([df_tier1, df_tier2, df_tier3], ignore_index=True)

    # Define dataset order and labels
    datasets = ["torres_torres", "ibarra_zarate", "raeisi", "wang"]
    dataset_labels = ["A", "B", "C", "D"]
    
    # Define model order and labels
    models = ["elastic_net", "random_forest", "svm", "eegnet"]
    model_labels = ["Tier 1 · elastic net", "Tier 2 · RF", "Tier 2 · SVM", "Tier 3 · EEGNet"]
    
    # Define metrics
    metrics = ["balanced_accuracy", "roc_auc"]
    
    # Create figure with 2 rows and 4 columns
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Pairwise cross-dataset transfer (12 ordered pairs)")
    
    # Process each metric and model combination
    for i, metric in enumerate(metrics):
        for j, model in enumerate(models):
            ax = axes[i, j]
            
            # Initialize 4x4 matrix
            matrix = np.full((4, 4), np.nan)
            
            # Fill the matrix with values
            for _, row in df.iterrows():
                if row["model"] == model:
                    try:
                        source_idx = datasets.index(row["source"])
                        target_idx = datasets.index(row["target"])
                        
                        # Skip the diagonal (self-transfer)
                        if source_idx != target_idx:
                            matrix[source_idx, target_idx] = row[metric]
                    except ValueError:
                        # Skip invalid dataset names
                        continue
            
            # Plot heatmap
            im = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0)
            
            # Add numeric annotations to each cell
            for si in range(4):
                for ti in range(4):
                    val = matrix[si, ti]
                    if not np.isnan(val):
                        text_color = "white" if val < 0.5 else "black"
                        ax.text(ti, si, f"{val:.2f}", ha="center", va="center",
                                fontsize=8, color=text_color)
            
            # Set ticks and labels
            ax.set_xticks(range(4))
            ax.set_yticks(range(4))
            ax.set_xticklabels(dataset_labels)
            ax.set_yticklabels(dataset_labels)
            
            # Set labels
            if i == 0:
                ax.set_title(model_labels[j])
            if j == 0:
                ax.set_ylabel("source")
            if i == 1:
                ax.set_xlabel("target")
            
            # Add colorbar to the last subplot
            if i == 1 and j == 3:
                fig.colorbar(im, ax=ax)
    
    # Adjust layout
    fig.tight_layout()
    
    # Save figure
    outdir = FIGURES_DIR / "tier23"
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "pairwise_transfer_matrix.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    print(f"Saved figure to {path}")

if __name__ == "__main__":
    main()