"""Phase 3 PLZC complexity extraction using neurokit2."""

import numpy as np
import mne
import neurokit2 as nk


def permutation_transform(signal: np.ndarray, motif_length: int, time_delay: int) -> np.ndarray:
    """Transform a 1-D signal into an ordinal-motif sequence via neurokit2 (permutation transform)."""
    _, info = nk.complexity_ordinalpatterns(signal, delay=time_delay, dimension=motif_length)
    return info['Uniques']


def compute_plzc(signal: np.ndarray, motif_length: int, time_delay: int) -> float:
    """Compute PLZC for one 1-D signal via neurokit2 (ordinal motifs then LZ76 compression)."""
    plzc, _ = nk.complexity_plzc(signal, delay=time_delay, dimension=motif_length)
    return float(plzc)


def plzc_columns(channels: list[str]) -> list[str]:
    """Return plzc_<channel> column names."""
    return [f"plzc_{ch}" for ch in channels]


def extract_plzc(epochs: mne.Epochs, config: dict) -> tuple[np.ndarray, list[str]]:
    """Compute per-epoch per-channel PLZC; returns (features (n_epochs, n_channels), column_names)."""
    m = int(config['plzc']['motif_length'])
    tau = int(config['plzc']['time_delay'])
    data = epochs.get_data()  # (n_epochs, n_channels, n_samples)
    features = np.empty((data.shape[0], data.shape[1]))
    
    for e in range(data.shape[0]):
        for c in range(data.shape[1]):
            features[e, c] = compute_plzc(data[e, c, :], m, tau)
    
    columns = plzc_columns(epochs.ch_names)
    return (features, columns)