"""Loads minimally-processed epoch time series for EEGNet (Tier 3), returning (X, y, subject_ids) arrays plus a torch Dataset wrapper."""

import csv
import numpy as np
import torch
from torch.utils.data import Dataset
from _common import INVENTORY_CSV
from load_epochs import load_dataset_epochs

DATASET_KEYS = ("torres_torres", "ibarra_zarate", "raeisi", "wang")


def load_group_map() -> dict[str, str]:
    """Load subject_id -> Group mapping from inventory.csv."""
    with INVENTORY_CSV.open("r") as f:
        reader = csv.DictReader(f)
        return {row["Subject_ID"]: row["Group"] for row in reader}


def build_dataset(dataset_key: str):
    """Load epoch time series for one dataset, return (X, y, subject_ids) arrays."""
    group_map = load_group_map()
    epochs_by_subject = load_dataset_epochs(dataset_key)
    
    xs = []
    ys = []
    subjs = []
    
    for subject_id, epochs in sorted(epochs_by_subject.items()):
        # Get data and cast to float32
        data = epochs.get_data().astype(np.float32)  # shape (n_epochs, n_channels, n_times)
        
        # Handle label
        if subject_id not in group_map:
            raise KeyError(f"Subject {subject_id} not found in group map")
        label = 1 if group_map[subject_id] == "Tinnitus" else 0
        
        # Create labels and subject IDs for this subject's epochs
        y_sub = np.full(data.shape[0], label, dtype=np.int64)
        subj_sub = np.full(data.shape[0], subject_id, dtype=object)
        
        # Collect for concatenation
        xs.append(data)
        ys.append(y_sub)
        subjs.append(subj_sub)
    
    # Concatenate all subjects in order
    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys)
    subject_ids = np.concatenate(subjs)
    
    return X, y, subject_ids


class EpochTensorDataset(Dataset):
    """Wrapper to convert numpy arrays to PyTorch tensors."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx])


def main() -> None:
    """Sanity check: load one dataset and print shapes and counts."""
    dataset_key = "torres_torres"
    X, y, subject_ids = build_dataset(dataset_key)
    
    print(f"Dataset: {dataset_key}")
    print(f"X.shape: {X.shape}")
    print(f"y.shape: {y.shape}")
    print(f"subject_ids.shape: {subject_ids.shape}")
    print(f"Unique subjects: {len(np.unique(subject_ids))}")
    print(f"Class counts: {np.bincount(y)}")


if __name__ == "__main__":
    main()