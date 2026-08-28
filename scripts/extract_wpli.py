"""Phase 3 wPLI functional connectivity extraction using mne-connectivity."""

import numpy as np
import mne
from mne_connectivity import spectral_connectivity_epochs


def compute_wpli(epochs: mne.Epochs, config: dict) -> tuple[np.ndarray, np.ndarray]:
    """Compute subject-level wPLI (across all epochs) for all channel pairs; returns (wpli (n_nodes, n_nodes, n_freqs), freqs)."""
    sfreq = epochs.info['sfreq']
    bands = config['frequency_bands']
    fmin = min(b[0] for b in bands.values())
    fmax = max(b[1] for b in bands.values())
    con = spectral_connectivity_epochs(epochs, method='wpli', sfreq=sfreq, fmin=fmin, fmax=fmax, verbose=False)
    return (con.get_data(output='dense'), con.freqs)


def aggregate_wpli_bands(wpli: np.ndarray, freqs: np.ndarray, config: dict) -> tuple[np.ndarray, list[str]]:
    """Aggregate per-frequency wPLI into the 5 canonical bands (mean within band); returns (wpli_band (n_pairs, 5), band_names)."""
    # Extract unique channel pairs (lower triangular, off-diagonal)
    n_nodes = wpli.shape[0]
    pairs = [(i, j) for i in range(n_nodes) for j in range(i)]  # (larger_idx, smaller_idx), row-major over LOWER triangle
    wpli_pairs = np.array([wpli[i, j, :] for (i, j) in pairs])  # Shape: (n_pairs, n_freqs)

    # Get band information
    bands = config['frequency_bands']
    band_names = list(bands.keys())

    # Initialize result array
    wpli_band = np.empty((wpli_pairs.shape[0], len(band_names)))

    # Convert freqs to numpy array for comparisons
    freqs_array = np.array(freqs)

    # Aggregate wPLI for each band
    for j, name in enumerate(band_names):
        lo, hi = bands[name]
        mask = (freqs_array >= lo) & (freqs_array < hi)
        wpli_band[:, j] = wpli_pairs[:, mask].mean(axis=-1)

    return wpli_band, band_names


def wpli_columns(channels: list[str], band_names: list[str]) -> list[str]:
    """Generate wPLI column names in pair-major order."""
    columns = []
    for i in range(len(channels)):
        for j in range(i):
            for band in band_names:
                columns.append(f"wpli_{band}_{channels[j]}_{channels[i]}")
    return columns


def extract_wpli(epochs: mne.Epochs, config: dict) -> tuple[np.ndarray, list[str]]:
    """Compute subject-level wPLI features broadcast to every epoch; returns (features (n_epochs, n_pairs*5), column_names)."""
    wpli, freqs = compute_wpli(epochs, config)
    wpli_band, band_names = aggregate_wpli_bands(wpli, freqs, config)  # shape (n_pairs, 5)
    subject_vec = wpli_band.flatten()  # (n_pairs*5,), pair-major (band inner)
    n_epochs = len(epochs)
    features = np.tile(subject_vec, (n_epochs, 1))  # (n_epochs, n_pairs*5)
    columns = wpli_columns(epochs.ch_names, band_names)
    return features, columns