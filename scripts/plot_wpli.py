"""
Generate verification figures showing wPLI connectivity matrices for all datasets.

This script creates heatmap visualizations of wPLI connectivity matrices for each
dataset in the alpha frequency band (8.0-13.0 Hz), averaged across epochs.
Four figures are generated, one per dataset.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from _common import load_config, PROJECT_ROOT
from load_epochs import load_dataset_epochs
from extract_wpli import compute_wpli


def main():
    # Load config
    config = load_config()
    
    # Define output directory
    output_dir = PROJECT_ROOT / "specs" / "tinnitus-eeg-interpretability" / "phase3" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Dataset keys in the specified order
    dataset_keys = ["torres_torres", "ibarra_zarate", "raeisi", "wang"]
    
    for dataset_key in dataset_keys:
        # Load epochs for the dataset
        epochs_dict = load_dataset_epochs(dataset_key)
        
        # Pick the first subject deterministically
        subject_ids = sorted(epochs_dict.keys())
        subject_id = subject_ids[0]
        epochs = epochs_dict[subject_id]
        
        # Compute wPLI
        wpli, freqs = compute_wpli(epochs, config)
        
        # Extract alpha band (8.0-13.0 Hz)
        lo, hi = config['frequency_bands']['alpha']
        # Convert freqs to numpy array for proper comparison
        freqs_array = np.array(freqs)
        mask = (freqs_array >= lo) & (freqs_array < hi)
        band_matrix = wpli[:, :, mask].mean(axis=-1)  # Shape (n_channels, n_channels)
        
        # Symmetrize the matrix (since it's stored in lower triangle only)
        sym = band_matrix + band_matrix.T
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(sym, cmap='viridis', vmin=0.0, vmax=1.0)
        
        # Set ticks and labels
        channels = epochs.ch_names
        tick_positions = np.arange(len(channels))
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xticklabels(channels, rotation=90)
        ax.set_yticklabels(channels)
        
        # Add grid
        ax.grid(True, color='white', linewidth=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label("wPLI")
        
        # Set labels and title
        ax.set_xlabel("Channel")
        ax.set_ylabel("Channel")
        title = f"{dataset_key} — {subject_id} wPLI (alpha band)"
        ax.set_title(title)
        
        # Save figure
        filename = f"4.4_{dataset_key}_{subject_id}_wpli_alpha.png"
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Print progress
        print(f"{dataset_key} {subject_id} -> {filename}")


if __name__ == "__main__":
    main()