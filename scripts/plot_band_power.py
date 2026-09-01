"""
Script to generate verification figures showing mean band power spectra across epochs for one sample subject per dataset.

This script creates figures that visualize the Welch band power extraction for verification purposes.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _common import load_config, PROJECT_ROOT
from load_epochs import load_dataset_epochs
from extract_power import compute_welch_psd

def main():
    config = load_config()
    frequency_bands = config['frequency_bands']
    
    # Define colors for each band
    band_colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightpink', 'lightcoral']
    
    # Define dataset keys in the specified order
    dataset_keys = ['torres_torres', 'ibarra_zarate', 'raeisi', 'wang']
    
    # Create figures directory
    figures_dir = PROJECT_ROOT / "specs" / "tinnitus-eeg-interpretability" / "phase3" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    for dataset_key in dataset_keys:
        # Get epochs for the dataset
        epochs_dict = load_dataset_epochs(dataset_key)
        
        # Pick the first subject deterministically
        subject_ids = sorted(epochs_dict.keys())
        subject_id = subject_ids[0]
        
        print(f"Processing {dataset_key} with subject {subject_id}")
        
        # Load epochs for the selected subject
        epochs = epochs_dict[subject_id]
        
        # Compute Welch PSD
        psd, freqs = compute_welch_psd(epochs, config)
        
        # Take the mean across epochs
        psd_mean = psd.mean(axis=0)  # shape (n_channels, n_freqs)
        
        # Convert to decibels
        db = 10 * np.log10(np.maximum(psd_mean, 1e-30))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot individual channel power spectra (faint gray lines)
        for ch in range(db.shape[0]):
            ax.plot(freqs, db[ch], alpha=0.25, linewidth=1, color='gray')
        
        # Plot mean across channels (bold black line)
        mean_power = db.mean(axis=0)
        ax.plot(freqs, mean_power, linewidth=2, color='black')
        
        # Shade frequency bands
        for (band_name, (lo, hi)), color in zip(frequency_bands.items(), band_colors):
            ax.axvspan(lo, hi, color=color, alpha=0.15)
            # Add band name label near the top of the axis
            ax.text((lo + hi) / 2, ax.get_ylim()[1] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.1,
                    band_name, ha='center', va='top', fontsize=10)
        
        # Set labels and title
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power (dB)")
        ax.set_xlim(min(frequency_bands.values())[0], max(frequency_bands.values())[1])
        ax.set_title(f"{dataset_key} — {subject_id} band power")
        ax.grid(True, alpha=0.3)
        
        # Save figure
        filename = f"3.4_{dataset_key}_{subject_id}_band_power.png"
        filepath = figures_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"{dataset_key} {subject_id} -> {filename}")

if __name__ == "__main__":
    main()