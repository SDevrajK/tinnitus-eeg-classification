"""
Generates Tier 3 spatial-filter topomaps and saliency heatmaps with proper channel/topography mappings.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import mne
from mne.channels import make_standard_montage
from _common import RESULTS_DIR, FIGURES_DIR, load_shared_channels

def main():
    # Load the spatial filters and saliency data
    spatial_filters = torch.load(RESULTS_DIR / "eegnet_spatial_filters.pt", weights_only=False)
    saliency = torch.load(RESULTS_DIR / "eegnet_saliency.pt", weights_only=False)
    
    # Load shared channels
    shared_channels = load_shared_channels()
    
    # Build MNE topomap support once
    montage = make_standard_montage("standard_1020")
    info = mne.create_info(ch_names=shared_channels, sfreq=256, ch_types="eeg")
    info.set_montage(montage)
    
    # Dataset keys with display labels
    dataset_keys = ["torres_torres", "ibarra_zarate", "raeisi", "wang"]
    dataset_labels = ["Dataset A", "Dataset B", "Dataset C", "Dataset D"]
    
    # FIGURE 1 — spatial filters
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    axes = axes.flatten()  # Flatten for easier indexing
    
    for i, (dataset_key, label) in enumerate(zip(dataset_keys, dataset_labels)):
        ax = axes[i]
        # Compute the per-channel spatial-filter magnitude
        values = np.abs(spatial_filters[dataset_key]).mean(axis=0)  # shape (13,)
        # Render a topomap
        mne.viz.plot_topomap(values, info, axes=ax, show=False, cmap="RdBu_r")
        ax.set_title(label)
    
    fig.suptitle("EEGNet-8,2 spatial filter magnitude (mean |weight| per channel)")
    fig.tight_layout()
    
    # Save spatial filter figure
    outdir = FIGURES_DIR / "tier3"
    outdir.mkdir(parents=True, exist_ok=True)
    path1 = outdir / "eegnet_spatial_filters.png"
    fig.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    print(f"Saved: {path1}")
    
    # FIGURE 2 — saliency maps
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()  # Flatten for easier indexing
    
    for i, (dataset_key, label) in enumerate(zip(dataset_keys, dataset_labels)):
        ax = axes[i]
        # Create heatmap
        im = ax.imshow(saliency[dataset_key], aspect="auto", origin="lower", cmap="hot")
        ax.set_yticks(range(13))
        ax.set_yticklabels(shared_channels)
        ax.set_xlabel("time (samples)")
        ax.set_ylabel("channel")
        # Add colorbar
        fig.colorbar(im, ax=ax)
        ax.set_title(label)
    
    fig.suptitle("EEGNet-8,2 Integrated-Gradients saliency (|attribution|, averaged over epochs)")
    fig.tight_layout()
    
    # Save saliency figure
    path2 = outdir / "eegnet_saliency.png"
    fig.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    print(f"Saved: {path2}")
    print(f"Output paths: {path1}, {path2}")

if __name__ == "__main__":
    main()