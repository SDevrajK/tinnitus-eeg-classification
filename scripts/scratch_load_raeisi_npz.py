#!/usr/bin/env python3
"""
Scratch loader: pull one Dataset C (Raeisi) resting-state .npz file into MNE.

The .npz files contain a single array `arr_0` of shape (63, 360000) recorded at
1200 Hz (300 s of continuous, Cz-referenced EEG). This script wraps that array
in an mne.io.RawArray so the data can be inspected (time series, PSD, etc.).

Run with `python -i` so the `raw` object stays in scope for poking around:
    python -i scripts/scratch_load_raeisi_npz.py
"""

import numpy as np
import mne
from pathlib import Path

# --- Config ---
RAEISI_FILE = Path(
    "/home/sdevrajk/media-hdd/researchdata/personal/MachineLearning/data/raw/dataset_C_raeisi/T1.npz"
)
RAEISI_SAMPLING_RATE = 1200.0

# The 64 channel names as listed in the Zenodo README (10-10 system), in order.
# The deposited files hold only 63 channels; Cz is the recorded reference
# ("Reference Channel") and is dropped here. This row->name mapping is our best
# inference from the README and has not been confirmed with the depositor.
RAEISI_CHANNEL_NAMES_64 = [
    "FP1", "FPz", "FP2", "AF7", "AF3", "AFz", "AF4", "AF8",
    "F9", "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8", "F10",
    "FT9", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8", "FT10",
    "T9", "T7", "C5", "C3", "C1",
    "Cz",  # reference, dropped (not present in the 63-channel file)
    "C2", "C4", "C6", "T8", "T10",
    "TP9", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8", "TP10",
    "P9", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", "P10",
    "POz",
]
REFERENCE_CHANNEL = "Cz"


def load_raeisi_file(file_path: Path) -> mne.io.RawArray:
    """Load a Dataset C .npz file into an mne.io.RawArray with 10-10 channel names."""
    data = np.load(file_path)["arr_0"]  # shape (63, 360000)
    data = np.asarray(data, dtype=np.float64)  # RawArray wants float64

    channel_names = [name for name in RAEISI_CHANNEL_NAMES_64 if name != REFERENCE_CHANNEL]
    if len(channel_names) != data.shape[0]:
        raise ValueError(
            f"Expected {data.shape[0]} channel names, got {len(channel_names)}"
        )

    info = mne.create_info(
        ch_names=channel_names,
        sfreq=RAEISI_SAMPLING_RATE,
        ch_types="eeg",
    )
    raw = mne.io.RawArray(data, info, verbose=False)

    # Best-effort montage. NOTE: MNE uses "Fp1/Fpz/Fp2" (lowercase p) while the
    # README uses "FP1/FPz/FP2", so those three will warn as unmatched.
    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage, on_missing="warn")

    return raw


if __name__ == "__main__":
    raw = load_raeisi_file(RAEISI_FILE)
    print(raw.info)
    print(f"\nLoaded {RAEISI_FILE.name}: {len(raw.ch_names)} channels, "
          f"{raw.info['sfreq']:.0f} Hz, {raw.times[-1]:.0f} s")

    # Poke around below (uncomment as needed):
    # raw.plot(duration=5, scalings="auto")
    # raw.plot_psd(fmax=60)
    # raw.compute_psd().plot_topomap(normalize=True)
