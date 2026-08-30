"""
Phase 2 raw-EEG preprocessing for Datasets A/B/C.

This script handles the loading and standardization of raw EEG data from
Datasets A, B, and C. It applies channel name mapping from config and selects
only shared channels.

Artifact-rejection approach: bad-channel detection + epoch rejection (not ASR).
This follows standard band-pass filtering and epoching practice (DOI: 10.3389/fninf.2015.00026).
"""

from pathlib import Path
import numpy as np
import mne
import csv
from eeg_processor.processing.filtering import filter_data
from eeg_processor.processing.artifact import detect_bad_channels
from eeg_processor.processing.reject_epochs import reject_bad_epochs

from _common import BIDS_ROOT, DERIVATIVES_DIR, PREPROCESSED_DIR, load_config, load_shared_channels

# Module-level constants
RAW_DATASET_KEYS = ("torres_torres", "ibarra_zarate", "raeisi")  # parent 4 handles raw datasets only


def load_excluded_montage(config) -> dict:
    """Load the montage exclusion map from config."""
    return config.get('excluded_subjects_montage', {})


def load_excluded_duplicates(config) -> dict:
    """Load the duplicate-recording exclusion map from config."""
    return config.get('excluded_subjects_duplicates', {})


def load_subject(dataset_key: str, subject_id: str, config: dict) -> mne.io.Raw:
    """
    Load and standardize raw EEG data for a single subject from a specific dataset.
    
    Parameters:
        dataset_key (str): The dataset identifier ('torres_torres', 'ibarra_zarate', or 'raeisi')
        subject_id (str): The subject ID (e.g., 'P01GA')
        config (dict): The loaded configuration dictionary
    
    Returns:
        mne.io.Raw: The preloaded, standardized, and channel-picked raw data
    """
    # Locate the data file
    dataset_root = BIDS_ROOT / dataset_key / f"sub-{subject_id}" / "eeg"
    
    # Find the EEG file (ignore .fdt files)
    if dataset_key == "raeisi":
        # Dataset C uses .npz files
        eeg_file = list(dataset_root.glob("*.npz"))[0]
        raw = _load_raeisi(eeg_file, config)
    else:
        # Datasets A and B use .set files
        eeg_file = list(dataset_root.glob("*.set"))[0]
        raw = _load_eeglab_file(eeg_file, config, dataset_key)
    
    # Select only shared channels
    shared_channels = load_shared_channels()
    raw.pick_channels(shared_channels)
    
    # Set channel types to EEG
    # (Assuming they're already EEG, but explicitly ensure)
    raw.set_channel_types({ch: 'eeg' for ch in raw.ch_names})
    
    # Set standard montage for bad-channel interpolation
    raw.set_montage(mne.channels.make_standard_montage('standard_1020'), verbose=False)
    
    return raw


def _load_eeglab_file(eeg_file: Path, config: dict, dataset_key: str) -> mne.io.Raw:
    """Load EEG data from EEGLAB .set file and apply dataset-specific preprocessing."""
    raw = mne.io.read_raw_eeglab(eeg_file, preload=True, verbose=False)
    
    if dataset_key == "ibarra_zarate":
        # Handle sampling rate normalization (512 -> 256 Hz)
        if raw.info['sfreq'] == 512.0:
            raw.resample(256.0)
        
        # Handle eyes-open split
        if len(raw.annotations) > 0:
            # Use first annotation, which should mark eyes-open/eyes-closed transition
            onset = float(raw.annotations.onset[0])
            split_t = onset if 60.0 < onset < raw.times[-1] else min(180.0, raw.times[-1])
        else:
            # Default to 3 minute split (180 seconds)
            split_t = min(180.0, raw.times[-1])
        
        # Keep only the eyes-open first ~3 mins
        raw = raw.crop(tmin=0.0, tmax=split_t)
        
        # Rename channels using the config mapping
        raw.rename_channels(config['channel_mapping']['ibarra_zarate'], on_missing='ignore')
    else:
        # Dataset A (torres_torres)
        # Rename channels using the config mapping
        raw.rename_channels(config['channel_mapping']['torres_torres'], on_missing='ignore')
    
    return raw


def _load_raeisi(eeg_file: Path, config: dict) -> mne.io.Raw:
    """Load RAEISI dataset from .npz file and create Raw object."""
    # Load the .npz file
    data = np.load(eeg_file, mmap_mode='r')
    arr = data['arr_0']  # Shape: 63 x n_samples (float64)
    
    # Trim the digitizer-startup artifact before ANY downstream processing.
    # Each raeisi file begins with a run of zero samples (exactly 0.0 in the
    # float64 files; ±0.01 quantization noise in the float32 re-saves) followed
    # by a single-sample spike whose magnitude exceeds any EEG value. That spike
    # is the front-end's first live sample while its DC-coupled input settles.
    # We remove everything up to and including the spike so that neither the
    # anti-aliasing low-pass in resample() nor the bandpass high-pass ever sees
    # a step discontinuity (which the high-pass would ring on, contaminating the
    # first ~2 s of the recording). The threshold is config-driven and applied
    # in raw .npz units (NOT µV) on the un-scaled array.
    threshold = float(config['preprocessing']['raeisi_trim_threshold'])
    exceed = (np.abs(arr) > threshold).any(axis=0)
    if not exceed.any():
        raise ValueError(
            f"{eeg_file.name}: no sample exceeds trim threshold {threshold}; "
            "refusing to trim (unexpected file structure)"
        )
    spike_idx = int(np.argmax(exceed))  # first sample where ANY channel exceeds threshold
    arr = arr[:, spike_idx + 1:]        # drop zeros (0..spike_idx-1) and the spike itself
    
    # Get the ordered 63-name list of canonical names
    names = config['channel_mapping']['raeisi']
    
    # Create the Raw object with canonical names
    # Note: .npz files store EEG in microvolts (µV), not volts (V)
    raw = mne.io.RawArray(arr.copy().astype(np.float64) / 1e6, mne.create_info(ch_names=names, sfreq=1200.0, ch_types='eeg'), verbose=False)
    
    # Downsample to the common target rate BEFORE any other preprocessing
    # (filtering/artifact-rejection/epoching), so the permutation time delay τ
    # (in samples) corresponds to a consistent physical lag across datasets.
    # mne.io.Raw.resample applies an anti-aliasing low-pass filter automatically.
    target_sfreq = float(config['preprocessing']['resample_sfreq'])
    if raw.info['sfreq'] != target_sfreq:
        raw.resample(target_sfreq)
    return raw


def main():
    """Main loader validation function - loads all subjects and prints summaries."""
    config = load_config()
    shared_channels = load_shared_channels()
    print(f"Loaded {len(shared_channels)} shared channels: {shared_channels}")
    
    # Load the montage exclusion map
    montage_excluded = load_excluded_montage(config)
    
    # Load the duplicate-recording exclusion map
    duplicate_excluded = load_excluded_duplicates(config)
    
    total_subjects = 0
    total_files_written = 0
    
    for dataset_key in RAW_DATASET_KEYS:
        print(f"\nProcessing dataset: {dataset_key}")
        
        # Discover subjects by globbing sub-* directories
        dataset_root = BIDS_ROOT / dataset_key
        subject_dirs = [d for d in dataset_root.iterdir() if d.is_dir()]
        subject_ids = [d.name.replace('sub-', '') for d in subject_dirs]
        
        print(f"Found {len(subject_ids)} subjects: {subject_ids}")
        
        for subject_id in subject_ids:
            total_subjects += 1
            
            # Check if subject should be excluded due to non-standard montage
            if subject_id in montage_excluded.get(dataset_key, []):
                print(f"  {dataset_key} {subject_id}: EXCLUDED (non-standard channel montage)")
                continue
            
            # Check if subject should be excluded as a duplicate recording
            if subject_id in duplicate_excluded.get(dataset_key, []):
                print(f"  {dataset_key} {subject_id}: EXCLUDED (duplicate recording)")
                continue
            
            # Compute output path
            out_path = PREPROCESSED_DIR / dataset_key / f"{subject_id}.fif"
            
            # Skip if already exists (makes runs resumable)
            if out_path.exists():
                print(f"  {dataset_key} {subject_id}: skip (exists)")
                total_files_written += 1
                continue
            
            try:
                raw = load_subject(dataset_key, subject_id, config)
                raw = preprocess_raw(raw, config)
                n_epochs = epoch_and_save(raw, dataset_key, subject_id, config)
                print(f"  {dataset_key} {subject_id}: {n_epochs} epochs, {raw.info['sfreq']:.0f} Hz")
                total_files_written += 1
            except Exception as e:
                print(f"  ERROR {dataset_key} {subject_id}: {e}")

    print(f"\nSummary: {total_subjects} subjects attempted, {total_files_written} .fif files written")
    
    # Write montage exclusion log
    write_montage_exclusion_log(montage_excluded)
    
    # Write duplicate-recording exclusion log
    write_duplicate_exclusion_log(duplicate_excluded)


# Stub functions for the next subtasks (4.2 and 4.3)
def preprocess_raw(raw: mne.io.Raw, config: dict) -> mne.io.Raw:
    """
    Filter + bad-channel detection via eeg-processor (subtask 4.2).
    
    This function applies bandpass filtering, notch filtering, and automated
    bad-channel detection + interpolation to raw EEG data.
    
    Parameters:
        raw (mne.io.Raw): The raw EEG data to preprocess
        config (dict): The loaded configuration dictionary
    
    Returns:
        mne.io.Raw: The filtered and cleaned raw data
        
    Note: This function is only invoked for raw (preprocessed=false) datasets.
    """
    # Read filter parameters from config
    l_freq = config['preprocessing']['l_freq']
    h_freq = config['preprocessing']['h_freq']
    notch_freq = config['preprocessing']['notch_freq']
    
    # Apply bandpass + notch filtering via eeg-processor
    raw = filter_data(raw, l_freq=l_freq, h_freq=h_freq, notch=notch_freq)
    
    # Apply automated bad-channel detection + interpolation via eeg-processor
    raw = detect_bad_channels(raw, interpolate=True, verbose=False)
    
    return raw


def epoch_and_save(raw: mne.io.Raw, dataset_key: str, subject_id: str, config: dict) -> int:
    """
    Fixed-length epochs + reject + save .fif (subtask 4.3).

    Parameters:
        raw (mne.io.Raw): The preprocessed raw EEG data
        dataset_key (str): The dataset identifier ('torres_torres', 'ibarra_zarate', or 'raeisi')
        subject_id (str): The subject ID (e.g., 'P01GA')
        config (dict): The loaded configuration dictionary

    Returns:
        int: The number of epochs saved after rejection
    """
    # Read epoch length from config
    epoch_len = config['preprocessing']['epoch_length_sec']
    
    # Create fixed-length epochs
    epochs = mne.make_fixed_length_epochs(raw, duration=epoch_len, preload=True, verbose=False)
    
    # Apply automated epoch-level artifact rejection via eeg-processor
    # Use data-adaptive rejection: reject epochs whose peak-to-peak exceeds
    # reject_ptp_median_multiple × the subject's median epoch PTP
    data = epochs.get_data()  # Shape: n_epochs x n_channels x n_samples
    max_ptp = (data.max(axis=2) - data.min(axis=2)).max(axis=1)  # Max PTP per epoch
    multiple = float(config['preprocessing']['reject_ptp_median_multiple'])
    reject_v = multiple * float(np.median(max_ptp))
    flat_v = config['preprocessing']['flat_eeg_v']
    
    # Guard against degenerate cases: if reject_v is not a positive finite number
    # (e.g. median PTP is 0), set reject_v = None so only flat rejection applies
    if not (np.isfinite(reject_v) and reject_v > 0):
        reject_v = None
        
    # Apply rejection (note: check_gradient=False since adaptive PTP threshold handles artifacts)
    epochs = reject_bad_epochs(epochs, reject={'eeg': reject_v} if reject_v is not None else None,
                               flat={'eeg': flat_v}, check_gradient=False, verbose=False)
    
    # Build the output path
    out_dir = PREPROCESSED_DIR / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{subject_id}.fif"
    
    # Save epochs
    epochs.save(out_path, overwrite=True)
    
    # Return the number of epochs saved
    return len(epochs)


def write_montage_exclusion_log(montage_excluded: dict):
    """Write a machine-readable log of montage-excluded subjects."""
    log_path = DERIVATIVES_DIR / "excluded_subjects_montage.csv"
    
    # Create the derivatives directory if needed
    DERIVATIVES_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(log_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['Subject_ID', 'Dataset', 'Reason'])
        writer.writeheader()
        
        for dataset_key, subjects in montage_excluded.items():
            for subject_id in subjects:
                writer.writerow({
                    'Subject_ID': subject_id,
                    'Dataset': dataset_key,
                    'Reason': 'non-standard channel montage'
                })


def write_duplicate_exclusion_log(duplicate_excluded: dict):
    """Write a machine-readable log of duplicate-recording-excluded subjects."""
    log_path = DERIVATIVES_DIR / "excluded_subjects_duplicates.csv"
    
    # Create the derivatives directory if needed
    DERIVATIVES_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(log_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['Subject_ID', 'Dataset', 'Reason'])
        writer.writeheader()
        
        for dataset_key, subjects in duplicate_excluded.items():
            for subject_id in subjects:
                writer.writerow({
                    'Subject_ID': subject_id,
                    'Dataset': dataset_key,
                    'Reason': 'duplicate recording'
                })


if __name__ == "__main__":
    main()