"""
Generate quality-control PSD figures for a sample of subjects per dataset and group.
"""

from pathlib import Path
import csv
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use non-interactive Agg backend for headless saving
import matplotlib.pyplot as plt
import mne

from _common import PROJECT_ROOT, PREPROCESSED_DIR, INVENTORY_CSV, load_config

FIGURES_DIR = PROJECT_ROOT / "specs" / "tinnitus-eeg-interpretability" / "phase2" / "figures"
DATASET_KEYS = ("torres_torres", "ibarra_zarate", "raeisi", "wang")


def load_group_map() -> dict[str, str]:
    """Load the subject-to-group mapping from inventory CSV."""
    group_map = {}
    with open(INVENTORY_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            group_map[row["Subject_ID"]] = row["Group"]
    return group_map


def select_qc_subjects(config: dict) -> list[dict]:
    """Select representative subjects per dataset and group for QC visualization."""
    n = int(config["qc"]["n_subjects_per_group"])
    group_map = load_group_map()
    
    # Collect all subjects by dataset and group
    dataset_group_subjects = defaultdict(list)
    
    for dataset_key in DATASET_KEYS:
        dataset_path = PREPROCESSED_DIR / dataset_key
        if not dataset_path.exists():
            continue
            
        for fif_file in dataset_path.glob("*.fif"):
            subject_id = fif_file.stem
            group = group_map.get(subject_id)
            if group is not None:
                dataset_group_subjects[(dataset_key, group)].append(subject_id)
    
    # Select first n subjects per dataset-group combination
    selected_subjects = []
    for (dataset_key, group), subject_ids in dataset_group_subjects.items():
        # Sort subject IDs alphabetically for deterministic selection
        subject_ids.sort()
        selected = subject_ids[:n]
        for subject_id in selected:
            fif_path = PREPROCESSED_DIR / dataset_key / f"{subject_id}.fif"
            selected_subjects.append({
                "dataset": dataset_key,
                "subject_id": subject_id,
                "group": group,
                "fif_path": fif_path
            })
    
    return selected_subjects


def plot_psd(epochs, dataset, subject_id, group, config) -> plt.Figure:
    """Plot PSD for the given epochs (subtask 6.2)."""
    # Read configuration values
    fmin = float(config['qc']['fmin'])
    fmax = float(config['qc']['fmax'])
    bands = config['qc']['bands']
    
    # Compute Welch PSD
    psd = epochs.compute_psd(method='welch', fmin=fmin, fmax=fmax, n_fft=int(round(epochs.info['sfreq'])), verbose=False)
    freqs = psd.freqs
    power = psd.get_data().mean(axis=0)  # average across epochs -> (n_channels, n_freqs)
    
    # Convert power to decibels 
    # Guard against zero with a small epsilon
    power = np.maximum(power, 1e-30)
    db = 10 * np.log10(power)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot each channel's spectrum as a faint gray line
    for channel_db in db:
        ax.plot(freqs, channel_db, alpha=0.25, linewidth=1)
    
    # Plot mean spectrum across channels as a bold dark line
    mean_db = np.mean(db, axis=0)
    ax.plot(freqs, mean_db, alpha=1, linewidth=2, color='black')
    
    # Shade each frequency band
    band_colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'lightgray']
    for (band_name, (lo, hi)), color in zip(bands.items(), band_colors):
        ax.axvspan(lo, hi, color=color, alpha=0.15)
        # Add centered text label for the band name near the top
        ax.text((lo + hi) / 2, ax.get_ylim()[1] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.1,
                band_name, ha='center', va='top', fontsize=10)
    
    # Add vertical dashed line at 50 Hz if applicable
    if fmax >= 50:
        ax.axvline(x=50, color='red', linestyle='--', alpha=0.7)
        ax.text(50, ax.get_ylim()[1] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.2,
                '50 Hz notch', ha='center', va='top', fontsize=10, color='red')
    
    # Set axes labels and title
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB)")
    ax.set_xlim(fmin, fmax)
    ax.set_title(f"{dataset} — {subject_id} ({group})")
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3)
    
    return fig


def save_figure(fig, dataset, subject_id, group) -> Path:
    """Save the figure to disk in FIGURES_DIR with descriptive filename."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{dataset}_{subject_id}_{group}_psd.png"
    filepath = FIGURES_DIR / filename
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    matplotlib.pyplot.close(fig)  # Free memory
    return filepath


def main():
    """Main function to generate and save QC figures for all selected subjects."""
    config = load_config()
    subjects = select_qc_subjects(config)
    
    saved_count = 0
    
    for subject in subjects:
        try:
            # Load the epochs
            epochs = mne.read_epochs(subject['fif_path'], verbose=False)
            
            # Plot PSD
            fig = plot_psd(epochs, subject['dataset'], subject['subject_id'], subject['group'], config)
            
            # Save figure
            saved = save_figure(fig, subject['dataset'], subject['subject_id'], subject['group'])
            
            # Print progress
            print(f"{subject['dataset']} {subject['subject_id']} ({subject['group']}) -> {saved.name}")
            
            saved_count += 1
            
        except Exception as e:
            print(f"ERROR {subject['dataset']} {subject['subject_id']}: {e}")
            continue
    
    print(f"Total figures saved: {saved_count}")
    print(f"Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()