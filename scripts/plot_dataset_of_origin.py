#!/usr/bin/env python3
"""
Dataset-of-origin confusion-matrix figure.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from _common import RESULTS_DIR, FIGURES_DIR

def main():
    # Load confusion matrix
    cm = pd.read_csv(RESULTS_DIR / "dataset_of_origin_rf_confusion_matrix.csv", index_col=0)
    cm_values = cm.values
    
    # Load balanced accuracy
    ba = pd.read_csv(RESULTS_DIR / "dataset_of_origin_rf_balanced_accuracy.csv")["balanced_accuracy"].iloc[0]
    
    # Dataset display labels
    dataset_labels = ["A", "B", "C", "D"]
    dataset_names = ["torres_torres", "ibarra_zarate", "raeisi", "wang"]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_values, cmap="Blues")
    
    # Annotate cells
    for i in range(len(dataset_names)):
        for j in range(len(dataset_names)):
            ax.text(j, i, f"{int(cm_values[i, j])}", 
                   ha="center", va="center", 
                   color="white" if cm_values[i,j] > cm_values.max()/2 else "black", 
                   fontsize=11)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(dataset_names)))
    ax.set_yticks(np.arange(len(dataset_names)))
    ax.set_xticklabels(dataset_labels)
    ax.set_yticklabels(dataset_labels)
    ax.set_xlabel("predicted dataset")
    ax.set_ylabel("true dataset")
    
    # Add colorbar
    fig.colorbar(im, ax=ax)
    
    # Set title
    ax.set_title(f"Dataset-of-origin classifier (Random Forest) — balanced accuracy = {ba:.3f}")
    
    # Adjust layout
    fig.tight_layout()
    
    # Save figure
    path = FIGURES_DIR / "tier23" / "dataset_of_origin_confusion_matrix.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    print(f"balanced accuracy = {ba:.4f}")
    print(f"Saved figure to {path}")

if __name__ == "__main__":
    main()