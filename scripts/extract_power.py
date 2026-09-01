"""Phase 3 band-power extraction: Welch PSD per epoch aggregated into 5 canonical frequency bands."""

import numpy as np
import mne

def compute_welch_psd(epochs: mne.Epochs, config: dict) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-epoch Welch PSD with configurable window/overlap; returns (psd, freqs)."""
    sfreq = epochs.info['sfreq']
    window_sec = config['welch_psd']['window_sec']
    overlap = config['welch_psd']['overlap']
    n_fft = int(round(window_sec * sfreq))
    n_overlap = int(round(overlap * n_fft))
    
    # Compute min and max band edges across all bands
    freq_bands = config['frequency_bands']
    fmin = min(band[0] for band in freq_bands.values())
    fmax = max(band[1] for band in freq_bands.values())
    
    psd = epochs.compute_psd(method='welch', fmin=fmin, fmax=fmax, n_fft=n_fft, n_overlap=n_overlap, verbose=False)
    return (psd.get_data(), psd.freqs)

def aggregate_bands(psd: np.ndarray, freqs: np.ndarray, config: dict) -> tuple[np.ndarray, list[str]]:
    """Aggregate PSD into the 5 canonical frequency bands; returns (band_power (n_epochs, n_channels, 5), band_names)."""
    bands = config['frequency_bands']
    band_names = list(bands.keys())
    band_power = np.empty((psd.shape[0], psd.shape[1], len(band_names)))
    
    for i, name in enumerate(band_names):
        lo, hi = bands[name]
        mask = (freqs >= lo) & (freqs < hi)
        band_power[:, :, i] = psd[:, :, mask].sum(axis=-1)
        
    return (band_power, band_names)

def band_power_columns(channels: list[str], band_names: list[str]) -> list[str]:
    """Generate column names for band power features in channel-major order."""
    return [f"power_{band}_{channel}" for channel in channels for band in band_names]

def extract_band_power(epochs: mne.Epochs, config: dict) -> tuple[np.ndarray, list[str]]:
    """Compute per-epoch band-power features; returns (features (n_epochs, n_channels*5), column_names)."""
    psd, freqs = compute_welch_psd(epochs, config)
    band_power, band_names = aggregate_bands(psd, freqs, config)
    features = band_power.reshape(band_power.shape[0], -1)  # flattens channel-major → (n_epochs, n_channels*5)
    columns = band_power_columns(epochs.ch_names, band_names)
    return (features, columns)