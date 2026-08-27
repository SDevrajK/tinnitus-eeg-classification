"""Phase 3 data loading: load epoched Phase 2 output for each dataset, restricted to shared channels."""

import mne
import numpy as np
from _common import PREPROCESSED_DIR, load_shared_channels


def get_shared_channels() -> list[str]:
    """Return the shared channel list from derivatives/shared_channels.json (via _common)."""
    return load_shared_channels()


def load_dataset_epochs(dataset_key: str, shared_channels: list[str] | None = None) -> dict[str, mne.Epochs]:
    """Load all epoched .fif files for one dataset, restricted to shared channels. Returns {subject_id: Epochs}."""
    if shared_channels is None:
        shared_channels = get_shared_channels()
    
    dataset_dir = PREPROCESSED_DIR / dataset_key
    epochs_dict = {}
    
    for fif_path in sorted(dataset_dir.glob("*.fif")):
        subject_id = fif_path.stem
        epochs = mne.read_epochs(fif_path, preload=True, verbose=False)
        epochs = epochs.pick_channels(shared_channels, ordered=True)
        epochs_dict[subject_id] = epochs
    
    return epochs_dict


def validate_epochs(epochs: mne.Epochs, expected_channels: list[str] | None = None) -> None:
    """Validate epochs structure (3D, shared-channel names/order) and finite values; raise ValueError on failure."""
    if expected_channels is None:
        expected_channels = get_shared_channels()
    
    data = epochs.get_data()
    
    if data.ndim != 3:
        raise ValueError("epochs data must be 3D (epochs × channels × samples)")
    
    if data.shape[1] != len(expected_channels):
        raise ValueError(f"epochs channel count {data.shape[1]} does not match expected {len(expected_channels)}")
    
    if list(epochs.ch_names) != expected_channels:
        raise ValueError("epochs channel names do not match expected shared channels")
    
    if not np.isfinite(data).all():
        raise ValueError("epochs data contains NaN or Inf")

