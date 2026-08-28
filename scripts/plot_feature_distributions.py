"""
Script to visualize feature distributions across all four datasets.

This script generates a figure comparing the distributions of power, wPLI, and PLZC
features across the four datasets: torres_torres, ibarra_zarate, raeisi, and wang.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _common import FEATURES_DIR, PROJECT_ROOT

def main():
    # Define the datasets in fixed order
    datasets = ['torres_torres', 'ibarra_zarate', 'raeisi', 'wang']
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Feature distributions across datasets")
    
    # Feature type configurations
    feature_configs = [
        {'type': 'power', 'prefix': 'power_', 'title': 'Band power (log10)', 'xlabel': 'log10 power'},
        {'type': 'wpli', 'prefix': 'wpli_', 'title': 'wPLI', 'xlabel': 'wPLI'},
        {'type': 'plzc', 'prefix': 'plzc_', 'title': 'PLZC', 'xlabel': 'PLZC'}
    ]
    
    # Process each feature type
    for i, config in enumerate(feature_configs):
        ax = axes[i]
        
        for dataset in datasets:
            # Load dataset
            df = pd.read_parquet(FEATURES_DIR / f"{dataset}_features.parquet")
            
            # Extract columns with the specified prefix
            feature_columns = [col for col in df.columns if col.startswith(config['prefix'])]
            
            # Flatten all values into a 1D array
            if config['type'] == 'power':
                # For power, flatten and apply log10 transformation
                vals = df[feature_columns].values.flatten()
                vals = np.log10(np.maximum(vals, 1e-30))
            else:
                # For wPLI and PLZC, just flatten
                vals = df[feature_columns].values.flatten()
            
            # Plot histogram
            ax.hist(vals, bins=50, density=True, alpha=0.5, label=dataset)
        
        # Set subplot properties
        ax.set_title(config['title'])
        ax.set_xlabel(config['xlabel'])
        ax.set_ylabel('density')
        
        # Add legend only to the first subplot
        if i == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save the figure
    output_path = PROJECT_ROOT / "specs" / "tinnitus-eeg-interpretability" / "phase3" / "figures" / "7.2_feature_distributions.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Figure saved to {output_path}")

if __name__ == "__main__":
    main()