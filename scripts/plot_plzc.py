"""
Plot PLZC distribution across channels for sample epochs.

This script generates boxplots showing the distribution of PLZC values across
channels for each dataset. Each figure displays one dataset with its first
subject's data.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from _common import load_config, PROJECT_ROOT
from load_epochs import load_dataset_epochs
from extract_plzc import extract_plzc

def main():
    # Load configuration
    config = load_config()
    
    # Define dataset keys in order
    dataset_keys = ["torres_torres", "ibarra_zarate", "raeisi", "wang"]
    
    # Create figures directory
    figures_dir = PROJECT_ROOT / "specs" / "tinnitus-eeg-interpretability" / "phase3" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    for dataset_key in dataset_keys:
        # Load epochs for the dataset
        epochs_dict = load_dataset_epochs(dataset_key)
        
        # Get the first subject ID (sorted)
        subject_ids = sorted(epochs_dict.keys())
        subject_id = subject_ids[0]
        
        # Load epochs for the first subject
        epochs = epochs_dict[subject_id]
        
        # Extract PLZC features
        features, columns = extract_plzc(epochs, config)
        
        # Create boxplot
        fig, ax = plt.subplots(figsize=(10, 6))
        channel_data = [features[:, c] for c in range(features.shape[1])]
        ax.boxplot(channel_data, tick_labels=epochs.ch_names)
        
        # Set labels and title
        ax.set_title(f"{dataset_key} — {subject_id} PLZC per channel")
        ax.set_ylabel("PLZC")
        ax.set_xlabel("Channel")
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, axis='y', alpha=0.3)
        
        # Save figure
        filename = f"5.6_{dataset_key}_{subject_id}_plzc.png"
        filepath = figures_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"{dataset_key} {subject_id} -> {filename}")

if __name__ == "__main__":
    main()