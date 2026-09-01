#!/usr/bin/env python3
"""
Naive-vs-corrected within-dataset CV comparison figure for Tiers 2 & 3.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from _common import RESULTS_DIR, FIGURES_DIR

def main():
    # Load the CSVs
    naive_df = pd.read_csv(RESULTS_DIR / "tier23_naive_cv.csv")
    corrected_df = pd.read_csv(RESULTS_DIR / "tier23_corrected_cv.csv")
    
    # Map dataset keys to display labels
    dataset_labels = {
        "torres_torres": "A",
        "ibarra_zarate": "B", 
        "raeisi": "C",
        "wang": "D"
    }
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Define model colors
    model_colors = {
        "random_forest": "#1f77b4",  # blue
        "svm": "#ff7f0e",           # orange
        "eegnet": "#2ca02c"         # green
    }
    
    # Define bar positions and width
    models = ["random_forest", "svm", "eegnet"]
    datasets = ["torres_torres", "ibarra_zarate", "raeisi", "wang"]
    x = np.arange(len(datasets))
    
    # Process each metric
    metrics = [("balanced_accuracy", ax1), ("roc_auc", ax2)]
    
    for metric_name, ax in metrics:
        # Create dictionaries to hold the data for each (dataset, model) pair
        naive_data = {}
        corrected_data = {}
        
        for dataset in datasets:
            for model in models:
                # Get naive value
                naive_row = naive_df[(naive_df["dataset"] == dataset) & (naive_df["model"] == model)]
                if not naive_row.empty:
                    naive_data[(dataset, model)] = naive_row[metric_name].values[0]
                
                # Get corrected value
                corrected_row = corrected_df[(corrected_df["dataset"] == dataset) & (corrected_df["model"] == model)]
                if not corrected_row.empty:
                    corrected_data[(dataset, model)] = corrected_row[metric_name].values[0]
        
        # Plot bars for each model
        for i, model in enumerate(models):
            naive_vals = [naive_data.get((dataset, model), np.nan) for dataset in datasets]
            corrected_vals = [corrected_data.get((dataset, model), np.nan) for dataset in datasets]
            
            # Fix bar positioning so that naive and corrected bars are side-by-side
            offset = (i - 1) * 0.25
            ax.bar(x + offset - 0.09/2, naive_vals, 0.09, label=f"{model} naive", color=model_colors[model], alpha=0.8)
            ax.bar(x + offset + 0.09/2, corrected_vals, 0.09, label=f"{model} corrected", color=model_colors[model], hatch='///', alpha=0.8)
        
        # Set axis properties
        ax.set_ylabel(metric_name)
        ax.set_xlabel("dataset")
        ax.set_xticks(x)
        ax.set_xticklabels([dataset_labels[ds] for ds in datasets])
        ax.set_title(metric_name)
        
        # Get y-axis limits based on data
        all_values = list(naive_data.values()) + list(corrected_data.values())
        all_values = [v for v in all_values if not np.isnan(v)]
        
        if all_values:
            y_min = min(all_values)
            y_max = max(all_values)
            ax.set_ylim(bottom=y_min - 0.05, top=y_max + 0.05)
        
        # Add grid for better readability
        ax.grid(axis='y', alpha=0.3)
    
    # Add legend
    handles, labels = ax1.get_legend_handles_labels()
    # Combine handles and labels from both axes and remove duplicates
    unique_handles_labels = []
    seen = set()
    for h, label in zip(handles, labels):
        if label not in seen:
            unique_handles_labels.append((h, label))
            seen.add(label)
    
    # Create the legend with all unique entries
    fig.legend([h for h, label in unique_handles_labels], [label for h, label in unique_handles_labels], 
               loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=6, frameon=True)
    
    # Add overall title
    fig.suptitle("Within-dataset CV: naive (leaky) vs corrected (subject-grouped) — Tiers 2 & 3")
    fig.tight_layout()
    
    # Save figure
    output_path = FIGURES_DIR / "tier23" / "naive_vs_corrected.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    print(f"Figure saved to: {output_path}")

if __name__ == "__main__":
    main()