"""
Tier 3 EEGNet LODO transfer (4 configurations).
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from _common import load_config, EPOCH_DROP_SEED, RANDOM_SEED, RESULTS_DIR
from lodo_pool import load_pool_epochs, cap_epochs_per_subject, align_n_times
from eegnet_data import build_dataset, EpochTensorDataset
from train_eegnet import build_model, train_one_epoch, balanced_class_weights


def main():
    config = load_config()
    eg = config["tier3"]["eegnet"]
    max_epochs = int(config["tier3"]["max_epochs"])
    batch_size = int(config["tier3"]["batch_size"])
    lr = float(config["tier3"]["learning_rate"])
    weight_decay = float(config["tier3"]["weight_decay"])
    class_weight = config["tier3"]["class_weight"]
    epoch_cap = int(config["phase4"]["epoch_cap"])
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    rows = []
    dataset_keys = ("torres_torres", "ibarra_zarate", "raeisi", "wang")
    
    for held_out in dataset_keys:
        # Build the pool
        X_pool, y_pool, subj_pool = load_pool_epochs(held_out, n_times=512)
        X_pool, y_pool, _ = cap_epochs_per_subject(X_pool, y_pool, subj_pool, epoch_cap, EPOCH_DROP_SEED)
        
        # Load + align the held-out test set
        X_test, y_test, _ = build_dataset(held_out)
        X_test = align_n_times(X_test, 512)
        
        # Train EEGNet on the pool
        model = build_model(X_pool.shape[1], X_pool.shape[2], eg).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Build the criterion with class weighting
        if class_weight == "balanced":
            w = balanced_class_weights(y_pool)
            criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32).to(device))
        else:
            criterion = nn.CrossEntropyLoss()
            
        train_ds = EpochTensorDataset(X_pool, y_pool)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        
        for epoch in range(max_epochs):
            train_one_epoch(model, train_loader, optimizer, criterion, device)
        
        # Evaluate on the held-out set in batches
        model.eval()
        y_probs = []
        test_ds = EpochTensorDataset(X_test, np.zeros(len(X_test), dtype=np.int64))
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        
        with torch.no_grad():
            for xb, _ in test_loader:
                y_prob = torch.softmax(model(xb.to(device)), dim=1)[:, 1]
                y_probs.append(y_prob.cpu().numpy())
        
        y_prob = np.concatenate(y_probs)
        y_pred = (y_prob >= 0.5).astype(int)
        ba = balanced_accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        row = {
            "held_out": held_out,
            "tier": "tier3",
            "model": "eegnet",
            "balanced_accuracy": ba,
            "roc_auc": auc
        }
        rows.append(row)
        print(f"Results for {held_out}: BA={ba:.4f}, AUC={auc:.4f}")
    
    # Write results
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "lodo_transfer_tier3.csv", index=False)


if __name__ == "__main__":
    main()