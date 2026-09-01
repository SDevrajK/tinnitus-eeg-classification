"""
Generates Tier 2 (RF + SVM) feature-importance bar-chart figures with physical feature labels.
"""

import pandas as pd
import matplotlib.pyplot as plt
from _common import load_config, RESULTS_DIR, FIGURES_DIR

def main():
    config = load_config()
    top_k = int(config["tier2"]["importance"]["top_k"])
    
    df = pd.read_csv(RESULTS_DIR / "tier2_top_features.csv")
    
    # Dataset order and labels
    dataset_keys = ("torres_torres", "ibarra_zarate", "raeisi", "wang")
    dataset_labels = ("Dataset A", "Dataset B", "Dataset C", "Dataset D")
    
    # Feature type color mapping
    feature_type_colors = {
        "power": "#1f77b4",  # matplotlib default blue
        "wpli": "#ff7f0e",   # matplotlib default orange
        "plzc": "#2ca02c"    # matplotlib default green
    }
    
    # Helper to construct human-readable labels
    def get_label(row):
        if row["feature_type"] == "power":
            return f"{row['band']} power · {row['channel']}"
        elif row["feature_type"] == "wpli":
            return f"{row['band']} wPLI · {row['channel']}"
        elif row["feature_type"] == "plzc":
            return f"PLZC · {row['channel']}"
        else:
            raise ValueError(f"Unknown feature_type: {row['feature_type']}")
    
    # Process each model
    for model in ("random_forest", "svm"):
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.flatten()
        
        # Get model name for title (replace underscores with spaces and title case)
        model_title = model.replace('_', ' ').title()
        
        # Process each dataset
        for i, (dataset_key, display_label) in enumerate(zip(dataset_keys, dataset_labels)):
            ax = axes[i]
            
            # Filter data for this model and dataset
            subset = df[(df["model"] == model) & (df["dataset"] == dataset_key)]
            
            # Sort by importance descending (stable sort to preserve rank order if tie)
            subset = subset.sort_values("importance", ascending=False, kind="stable")
            
            # Take top-k
            subset = subset.head(top_k)
            
            # Reverse to put highest importance at top (as per barh convention)
            subset = subset.iloc[::-1]
            
            # Extract physical labels and importance scores
            labels = [get_label(row) for _, row in subset.iterrows()]
            importances = subset["importance"].values
            
            # Draw horizontal bar chart
            ax.barh(labels, importances, color=[feature_type_colors[row["feature_type"]] for _, row in subset.iterrows()])
            
            # Set labels and title
            ax.set_title(f"{display_label} — {model_title}")
            ax.set_xlabel("importance")
            

            
            # Set font size for y ticks
            ax.tick_params(axis='y', labelsize=8)
        
        # Add a single legend for feature types
        legend_elements = [plt.Rectangle((0,0),1,1, color=feature_type_colors[ft]) for ft in ["power", "wpli", "plzc"]]
        fig.legend(legend_elements, ["power", "wPLI", "PLZC"], loc='upper center', bbox_to_anchor=(0.5, 0.96), ncol=3)
        
        # Set suptitle
        # Determine metric from model name
        if model == "random_forest":
            metric = "Gini feature importance"
        else:  # svm
            metric = "permutation feature importance"
        
        fig.suptitle(f"Tier 2 {model_title} — {metric} (top {top_k})")
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        
        # Save figure
        outdir = FIGURES_DIR / "tier2"
        outdir.mkdir(parents=True, exist_ok=True)
        path = outdir / f"{model}_feature_importance.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        print(f"Saved {path}")

if __name__ == "__main__":
    main()