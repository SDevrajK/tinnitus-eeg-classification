"""
Tier 3 EEGNet pairwise cross-dataset transfer (12 ordered pairs), reusing trained models;
aligns time length across 256/250 Hz datasets.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch.utils.data import DataLoader
from _common import load_config, RANDOM_SEED, RESULTS_DIR
from eegnet_data import build_dataset, EpochTensorDataset
from train_eegnet import build_model

def align_n_times(X: np.ndarray, n_times: int) -> np.ndarray:
    """Crop (if longer) or zero-pad (if shorter) the time axis to n_times."""
    if X.shape[2] == n_times:
        return X
    if X.shape[2] > n_times:
        return X[:, :, :n_times]
    pad = n_times - X.shape[2]
    return np.pad(X, ((0,0),(0,0),(0,pad)), mode="constant")

def predict_proba_batched(model, X, device, batch_size=512):
    model.eval()
    model.to(device)
    ds = EpochTensorDataset(X, np.zeros(len(X), dtype=np.int64))  # labels unused here
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    probs = []
    with torch.no_grad():
        for xb, _ in loader:
            out = model(xb.to(device))
            probs.append(torch.softmax(out, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probs)

def main():
    config = load_config()
    eg = config["tier3"]["eegnet"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    MODELS_DIR = RESULTS_DIR / "models"
    
    # Dataset keys, fixed order: ("torres_torres", "ibarra_zarate", "raeisi", "wang") = A, B, C, D.
    DATASET_KEYS = ("torres_torres", "ibarra_zarate", "raeisi", "wang")
    
    # Cache the per-dataset arrays so each dataset is loaded from disk ONCE (each appears as both source and target)
    data_cache = {}
    
    def get_data(key):
        if key not in data_cache:
            data_cache[key] = build_dataset(key)[:2]  # just X, y
        return data_cache[key]
    
    rows = []
    
    # For each ordered pair of datasets (12 total, no self-pairs)
    for source in DATASET_KEYS:
        for target in DATASET_KEYS:
            if source != target:
                # Load the pre-trained source model
                X_src, _ = get_data(source)
                n_chans, n_times = X_src.shape[1], X_src.shape[2]
                
                # Rebuild model and load state dict
                model = build_model(n_chans, n_times, eg)
                model.load_state_dict(torch.load(MODELS_DIR / f"{source}_eegnet.pt", weights_only=True))
                
                # Load and process target data
                X_tgt, y_tgt = get_data(target)
                X_tgt_a = align_n_times(X_tgt, n_times)
                
                # Predict using the source model on target data
                y_prob = predict_proba_batched(model, X_tgt_a, device)
                y_pred = (y_prob >= 0.5).astype(int)
                
                # Calculate metrics
                ba = balanced_accuracy_score(y_tgt, y_pred)
                auc = roc_auc_score(y_tgt, y_prob)
                
                # Store results
                rows.append({
                    "source": source,
                    "target": target,
                    "tier": "tier3",
                    "model": "eegnet",
                    "balanced_accuracy": ba,
                    "roc_auc": auc
                })
                print(f"{source} -> {target}: BA={ba:.4f} AUC={auc:.4f}")
    
    # Save results to CSV
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "pairwise_transfer_tier3.csv", index=False)

if __name__ == "__main__":
    main()