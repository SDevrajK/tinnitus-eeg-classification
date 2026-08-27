"""
Pipeline for already-preprocessed Dataset D (channel standardization + epoch preservation only).

This script loads epoched EEG data for Dataset D, which has already been preprocessed
and epoched at 2 seconds. The data cannot be read with standard MNE loaders due to
an empty event array in the .set file, so a custom loader is used.

Channel standardization is performed using the mapping defined in config.yaml.
"""
from pathlib import Path
import scipy.io
import numpy as np
import mne
import yaml
import csv
import json

# Constants
PROJECT_ROOT = Path("/home/sdevrajk/projects/personal/MachineLearning")
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
BIDS_ROOT = Path("/home/sdevrajk/media-hdd/researchdata/personal/MachineLearning/data/bids_organized")
DERIVATIVES_DIR = Path("/home/sdevrajk/media-hdd/researchdata/personal/MachineLearning/data/derivatives")
SHARED_CHANNELS_PATH = DERIVATIVES_DIR / "shared_channels.json"
PREPROCESSED_DIR = DERIVATIVES_DIR / "preprocessed"
EXCLUDED_CSV = DERIVATIVES_DIR / "excluded_subjects.csv"
DATASET_KEY = "wang"

def load_config() -> dict:
    """Load configuration from YAML file."""
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

def load_shared_channels() -> list[str]:
    """Load the list of shared channels from JSON."""
    with open(SHARED_CHANNELS_PATH, 'r') as f:
        data = json.load(f)
        # Extract the shared_channels list from the dictionary
        return data['shared_channels']

def load_excluded_subject_ids() -> set[str]:
    """Load excluded subject IDs from CSV file for Dataset D."""
    excluded = set()
    with open(EXCLUDED_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['Dataset'] == 'wang':
                excluded.add(row['Subject_ID'])
    return excluded

def load_wang_epochs(subject_id: str, config: dict) -> mne.Epochs:
    """
    Custom loader for epoched EEG data from Dataset D.
    
    Loads .set and .fdt files directly since MNE's EEGLAB readers fail on empty event arrays.
    Interpolates missing shared channels (e.g., T068 missing F3) using standard_1020 montage.
    """
    # Locate files
    eeg_dir = BIDS_ROOT / DATASET_KEY / f"sub-{subject_id}" / "eeg"
    set_file = next(eeg_dir.glob("*.set"))
    fdt_file = next(eeg_dir.glob("*.fdt"))
    
    # Load header
    mat = scipy.io.loadmat(str(set_file), struct_as_record=False, squeeze_me=True)
    eeg = mat["EEG"]
    nbchan = int(eeg.nbchan)
    trials = int(eeg.trials)
    pnts = int(eeg.pnts)
    sfreq = float(eeg.srate)
    labels = [c.labels for c in eeg.chanlocs]
    
    # Load data
    flat = np.fromfile(str(fdt_file), dtype=np.float32)
    data = flat.reshape((nbchan, pnts, trials), order='F')
    data = data.transpose(2, 0, 1)  # (trials, nbchan, pnts)
    data = data / 1e6  # Convert from microvolts to volts
    
    # Create epochs
    info = mne.create_info(ch_names=labels, sfreq=sfreq, ch_types='eeg')
    epochs = mne.EpochsArray(data, info, tmin=0.0, verbose=False)
    
    # Rename channels using mapping from config
    epochs.rename_channels(config['channel_mapping']['wang'], on_missing='ignore')
    
    # Pick only shared channels, interpolating missing ones
    shared = load_shared_channels()
    present = [c for c in shared if c in epochs.ch_names]
    missing = [c for c in shared if c not in epochs.ch_names]
    
    # Drop every channel not in the shared set
    epochs = epochs.pick_channels(present)
    
    # Build the montage once
    montage = mne.channels.make_standard_montage('standard_1020')
    
    if missing:  # Handle missing channels (e.g., T068 missing F3)
        # Add each missing channel as an all-zero channel
        n_epochs = len(epochs)
        n_times = epochs.get_data().shape[2]
        zeros = np.zeros((n_epochs, len(missing), n_times))
        missing_info = mne.create_info(missing, sfreq=epochs.info['sfreq'], ch_types='eeg')
        missing_epochs = mne.EpochsArray(zeros, missing_info, tmin=epochs.tmin, verbose=False)
        epochs = epochs.add_channels([missing_epochs], force_update_info=True)
        
        # Set the montage so the missing channel has a position
        epochs.set_montage(montage, on_missing='ignore', verbose=False)
        
        # Mark the missing channels bad and interpolate
        epochs.info['bads'] = list(missing)
        epochs.interpolate_bads(reset_bads=True, verbose=False)
    else:
        # Set the montage for all subjects for downstream use (when missing is empty)
        epochs.set_montage(montage, on_missing='ignore', verbose=False)
    
    # Enforce the canonical shared channel order
    epochs = epochs.reorder_channels(shared)
    
    return epochs

def save_wang_epochs(epochs, subject_id) -> int:
    """
    Save preprocessed epochs to .fif file.
    
    Preserves existing 2-s epochs (skip filtering and artifact rejection 
    per already_preprocessed flag), saves to derivatives/preprocessed/wang/
    
    Parameters
    ----------
    epochs : mne.Epochs
        Preprocessed epochs to save
    subject_id : str
        Subject ID for output file naming
        
    Returns
    -------
    int
        Number of epochs saved
    """
    # Create output directory
    out_dir = PREPROCESSED_DIR / DATASET_KEY
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save .fif file
    out_path = out_dir / f"{subject_id}.fif"
    epochs.save(out_path, overwrite=True)
    
    return len(epochs)

def main():
    """Main pipeline for Dataset D: load, standardize, and save preprocessed epochs."""
    config = load_config()
    shared_channels = load_shared_channels()
    excluded_subjects = load_excluded_subject_ids()
    
    # Find all subjects in Dataset D
    subject_dirs = (BIDS_ROOT / DATASET_KEY).glob("sub-*")
    subject_ids = [d.name.split("-")[1] for d in subject_dirs]
    
    print(f"Found {len(subject_ids)} subjects in Dataset D")
    
    total_processed = 0
    total_files = 0
    epoch_counts = []
    
    for subject_id in subject_ids:
        # Check if output file already exists
        out_path = PREPROCESSED_DIR / DATASET_KEY / f"{subject_id}.fif"
        if out_path.exists():
            print(f"{subject_id}: skip (exists)")
            continue
            
        if subject_id in excluded_subjects:
            print(f"{subject_id}: skipped (excluded)")
            continue
            
        try:
            # Load and process epochs
            epochs = load_wang_epochs(subject_id, config)
            n = save_wang_epochs(epochs, subject_id)
            print(f"{subject_id}: {n} epochs, {epochs.info['sfreq']:.0f} Hz")
            
            total_processed += 1
            total_files += 1
            epoch_counts.append(n)
            
        except Exception as e:
            print(f"{subject_id}: ERROR {e}")
            continue
    
    # Print summary
    print(f"\nSummary: Processed {total_processed} subjects, created {total_files} .fif files")
    if epoch_counts:
        print(f"Epoch count range: {min(epoch_counts)} - {max(epoch_counts)}, median: {np.median(epoch_counts):.0f}")

if __name__ == "__main__":
    main()