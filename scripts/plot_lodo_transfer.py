"""
LODO transfer-results figure.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from _common import RESULTS_DIR, FIGURES_DIR

def main():
    # Load the three tables into ONE unified DataFrame
    # Tier 1: read lodo_transfer.csv, add tier="tier1" and model="elastic_net" columns
    df_tier1 = pd.read_csv(RESULTS_DIR / "lodo_transfer.csv")
    df_tier1["tier"] = "tier1"
    df_tier1["model"] = "elastic_net"
    
    # Tier 2 and Tier 3: read their CSVs directly (they already have the unified columns)
    df_tier2 = pd.read_csv(RESULTS_DIR / "lodo_transfer_tier2.csv")
    df_tier3 = pd.read_csv(RESULTS_DIR / "lodo_transfer_tier3.csv")
    
    # Combine all dataframes
    df = pd.concat([df_tier1, df_tier2, df_tier3], ignore_index=True)
    
    # Reorder models for display 
    model_order = ["elastic_net", "random_forest", "svm", "eegnet"]
    
    # Map models to display names
    model_display = {
        "elastic_net": "Tier 1 · elastic net",
        "random_forest": "Tier 2 · RF",
        "svm": "Tier 2 · SVM", 
        "eegnet": "Tier 3 · EEGNet"
    }
    
    # Map held_out to display labels
    held_out_display = {
        "torres_torres": "A",
        "ibarra_zarate": "B",
        "raeisi": "C",
        "wang": "D"
    }
    
    # Set up the figure with 2 side-by-side panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Colors for each model
    colors = {
        "elastic_net": "#1f77b4",   # blue
        "random_forest": "#ff7f0e", # orange
        "svm": "#2ca02c",          # green
        "eegnet": "#d62728"        # red
    }
    
    # Get the unique held-out datasets
    held_out_datasets = df["held_out"].unique()
    
    # For each dataset, get all model results
    bar_width = 0.13
    group_width = 0.6
    group_positions = np.arange(len(held_out_datasets))
    
    # Process each panel (balanced_accuracy and roc_auc)
    metrics = ["balanced_accuracy", "roc_auc"]
    metric_labels = ["balanced accuracy", "AUC-ROC"]
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = ax1 if i == 0 else ax2
        
        # Plot grouped bars for each dataset
        for j, dataset in enumerate(held_out_datasets):
            dataset_data = df[df["held_out"] == dataset]
            for k, model in enumerate(model_order):
                if model in dataset_data["model"].values:
                    value = dataset_data[dataset_data["model"] == model][metric].iloc[0]
                    x_pos = group_positions[j] - group_width/2 + k * bar_width
                    ax.bar(x_pos, value, bar_width, 
                               label=model_display[model], color=colors[model], 
                               edgecolor='black', linewidth=0.2)
                    
                    # Add value annotation
                    ax.text(x_pos, value + 0.01, f"{value:.2f}", 
                           ha="center", va="bottom", fontsize=8)
        
        # Set labels and title
        ax.set_xticks(group_positions)
        ax.set_xticklabels([held_out_display[dataset] for dataset in held_out_datasets])
        ax.set_ylabel(label)
        ax.set_title(label)
        
        # Find appropriate y-limits (zoom to data)
        metric_values = df[metric]
        y_min, y_max = metric_values.min(), metric_values.max()
        y_range = y_max - y_min
        if y_range > 0:
            ax.set_ylim(bottom=y_min - 0.05, top=y_max + 0.05)
        else:
            ax.set_ylim(0, 1)
    
    # Add legend
    handles, labels = ax1.get_legend_handles_labels()
    # Remove duplicate legend entries
    unique_handles_labels = []
    seen_labels = set()
    for handle, label in zip(handles, labels):
        if label not in seen_labels:
            unique_handles_labels.append((handle, label))
            seen_labels.add(label)
    
    fig.legend([h[0] for h in unique_handles_labels], [h[1] for h in unique_handles_labels], 
              loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.02))
    
    # Set overall title
    fig.suptitle("Leave-one-dataset-out transfer (train on 3, test on held-out 1)")
    
    # Adjust layout
    fig.tight_layout(rect=[0, 0.12, 1, 0.95])
    
    # Save the figure
    output_path = FIGURES_DIR / "tier23" / "lodo_transfer.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    print(f"Saved figure to {output_path}")

if __name__ == "__main__":
    main()