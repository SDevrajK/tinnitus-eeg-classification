"""
Helper module for creating LODO training pools with subject-stratified epoch capping
and time-length alignment to handle class imbalance and dataset dominance issues.

The LODO evaluation uses 3-out-of-4 dataset configurations where one dataset is held out
and the other three are pooled for training. This module handles the imbalance/dataset-dominance
problem by:
1. Subject-stratified epoch capping to prevent any single subject from dominating the pool
2. Time-length alignment to allow heterogeneous datasets to be pooled for Tier 3
3. Class weighting applied by calling scripts (RF/SVM use class_weight="balanced", EEGNet uses balanced_class_weights)

Dataset keys in fixed order: ("torres_torres", "ibarra_zarate", "raeisi", "wang")
"""
import numpy as np
from _common import load_scaled_features
from eegnet_data import build_dataset

# Fixed dataset order
DATASET_KEYS = ("torres_torres", "ibarra_zarate", "raeisi", "wang")

# Feature column prefixes (copy from project convention)
FEATURE_PREFIXES = ("power_", "wpli_", "plzc_")

def feature_columns(df):
    """Return list of feature columns from a DataFrame."""
    return [col for col in df.columns if any(col.startswith(prefix) for prefix in FEATURE_PREFIXES)]

def cap_epochs_per_subject(X: np.ndarray, y: np.ndarray, subject_ids: np.ndarray, max_epochs: int, seed: int):
    """
    Deterministically limits each subject to at most max_epochs epochs by randomly dropping excess.
    
    Args:
        X: numpy array of shape (n_epochs, ...)
        y: numpy array of shape (n_epochs,)
        subject_ids: numpy array of shape (n_epochs,) with subject IDs
        max_epochs: maximum number of epochs per subject
        seed: random seed for reproducibility
        
    Returns:
        Tuple of (X_capped, y_capped, subject_ids_capped) with epoch capping applied
    """
    rng = np.random.default_rng(seed)
    keep_indices = []
    
    for subject_id in np.unique(subject_ids):
        these_indices = np.where(subject_ids == subject_id)[0]
        
        if len(these_indices) > max_epochs:
            # Randomly sample up to max_epochs without replacement
            selected_indices = rng.choice(these_indices, size=max_epochs, replace=False)
        else:
            # Keep all indices if we don't exceed max_epochs
            selected_indices = these_indices
            
        keep_indices.extend(selected_indices)
    
    # Sort indices to maintain consistent ordering
    keep_indices = np.sort(keep_indices)
    
    return X[keep_indices], y[keep_indices], subject_ids[keep_indices]

def align_n_times(X: np.ndarray, n_times: int) -> np.ndarray:
    """
    Crop or zero-pad the time axis to align to n_times.
    
    Args:
        X: numpy array of shape (n_epochs, n_channels, n_times)
        n_times: target number of time samples
        
    Returns:
        X aligned to n_times (cropped if longer, zero-padded if shorter)
    """
    current_n_times = X.shape[2]
    
    if current_n_times > n_times:
        # Crop to n_times
        return X[:, :, :n_times]
    elif current_n_times < n_times:
        # Pad with zeros
        pad_width = n_times - current_n_times
        return np.pad(X, ((0, 0), (0, 0), (0, pad_width)), mode='constant', constant_values=0)
    else:
        # Already correct length
        return X

def load_pool_scaled_features(held_out: str):
    """
    Load scaled features from the 3 datasets NOT equal to held_out.
    
    Args:
        held_out: dataset key to exclude (e.g., "torres_torres")
        
    Returns:
        Tuple of (X, y, subject_ids) where X is (n_epochs, 468)
    """
    # Determine which datasets to use (exclude held_out)
    datasets_to_use = [key for key in DATASET_KEYS if key != held_out]
    
    all_X = []
    all_y = []
    all_subject_ids = []
    
    for dataset_key in datasets_to_use:
        df = load_scaled_features(dataset_key)
        X = df[feature_columns(df)].to_numpy()  # float64 is fine
        y = (df["Group"]=="Tinnitus").astype(int).to_numpy()
        subject_ids = df["Subject_ID"].to_numpy()
        
        all_X.append(X)
        all_y.append(y)
        all_subject_ids.append(subject_ids)
    
    return (
        np.vstack(all_X),
        np.hstack(all_y),
        np.hstack(all_subject_ids)
    )

def load_pool_epochs(held_out: str, n_times: int = 512):
    """
    Load epoch data from the 3 datasets NOT equal to held_out.
    
    Args:
        held_out: dataset key to exclude (e.g., "torres_torres")
        n_times: target number of time samples for alignment (default 512)
        
    Returns:
        Tuple of (X, y, subject_ids) where X is (n_epochs, 13, n_times)
    """
    # Determine which datasets to use (exclude held_out)
    datasets_to_use = [key for key in DATASET_KEYS if key != held_out]
    
    all_X = []
    all_y = []
    all_subject_ids = []
    
    for dataset_key in datasets_to_use:
        X_k, y_k, subj_k = build_dataset(dataset_key)
        X_k = align_n_times(X_k, n_times)
        
        all_X.append(X_k)
        all_y.append(y_k)
        all_subject_ids.append(subj_k)
    
    return (
        np.vstack(all_X),
        np.hstack(all_y),
        np.hstack(all_subject_ids)
    )