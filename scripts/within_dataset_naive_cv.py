"""
Naive (leaky, epoch-level) within-dataset CV for Tiers 2 and 3.

This script runs within-dataset cross-validation using naive epoch-level stratified
K-fold splits (i.e., NOT grouped by subject). This deliberate leakage inflates
performance metrics compared to the corrected Leave-One-Subject-Out (LOSO) CV,
which is the purpose of quantifying this inflation in the project.

Why naive epoch-level splitting is leaky: epochs from the same subject share
subject-specific signal (notably, the wPLI features are literally identical across a
subject's epochs), so when a subject's epochs are split across train and test folds,
the classifier can partly learn subject identity rather than the tinnitus/control
class — inflating the score.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from _common import load_config, load_scaled_features, RANDOM_SEED, RESULTS_DIR
from eegnet_data import build_dataset, EpochTensorDataset
from train_eegnet import build_model, train_one_epoch, evaluate, balanced_class_weights

def main():
    config = load_config()
    eg = config["tier3"]["eegnet"]
    max_epochs = int(config["tier3"]["max_epochs"])
    batch_size = int(config["tier3"]["batch_size"])
    lr = float(config["tier3"]["learning_rate"])
    weight_decay = float(config["tier3"]["weight_decay"])
    class_weight = config["tier3"]["class_weight"]
    cv_folds = int(config["tier3"]["cv_folds"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Set seeds for reproducibility
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if device == "cuda":
        torch.cuda.manual_seed_all(RANDOM_SEED)

    # Build Tier 2 best-params maps from CSVs
    rf_best = {}
    rf_params = pd.read_csv(RESULTS_DIR / "random_forest_best_params.csv")
    for _, row in rf_params.iterrows():
        # Handle max_depth being None or NaN
        max_depth = row["best_max_depth"]
        if pd.isna(max_depth) or max_depth == "":
            max_depth = None
        else:
            max_depth = int(max_depth)
        
        rf_best[row["dataset"]] = (
            int(row["best_n_estimators"]),
            max_depth,
            int(row["best_min_samples_leaf"]),
            row["best_max_features"]  # This could be a string like "sqrt" or "log2"
        )
    
    svm_best = {}
    svm_params = pd.read_csv(RESULTS_DIR / "svm_best_params.csv")
    for _, row in svm_params.iterrows():
        gamma = row["best_gamma"]
        # Handle gamma being the string "scale"
        if pd.isna(gamma) or gamma == "":
            gamma = None
        svm_best[row["dataset"]] = (float(row["best_C"]), gamma)

    dataset_keys = ("torres_torres", "ibarra_zarate", "raeisi", "wang")
    rows = []

    for dataset_key in dataset_keys:
        print(f"\nComputing naive CV for dataset: {dataset_key}")
        
        # TIER 2 — Random Forest
        df = load_scaled_features(dataset_key)
        feat_cols = [col for col in df.columns if col.startswith(("power_", "wpli_", "plzc_"))]
        X = df[feat_cols].to_numpy()
        y = (df["Group"] == "Tinnitus").astype(int).to_numpy()
        
        rf_n_estimators, rf_max_depth, rf_min_samples_leaf, rf_max_features = rf_best[dataset_key]
        rf = RandomForestClassifier(
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            min_samples_leaf=rf_min_samples_leaf,
            max_features=rf_max_features,
            class_weight="balanced",
            random_state=RANDOM_SEED
        )
        
        # Naive epoch-level CV
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
        rf_ba_scores = []
        rf_auc_scores = []
        
        for train_idx, val_idx in skf.split(X, y):
            rf.fit(X[train_idx], y[train_idx])
            y_pred_proba = rf.predict_proba(X[val_idx])[:, 1]
            y_true = y[val_idx]
            
            rf_ba_scores.append(balanced_accuracy_score(y_true, y_pred_proba >= 0.5))
            rf_auc_scores.append(roc_auc_score(y_true, y_pred_proba))
        
        rf_ba = float(np.mean(rf_ba_scores))
        rf_auc = float(np.mean(rf_auc_scores))
        
        rows.append({
            "dataset": dataset_key,
            "tier": "tier2",
            "model": "random_forest",
            "balanced_accuracy": rf_ba,
            "roc_auc": rf_auc
        })
        print(f"  RF: BA={rf_ba:.4f} AUC={rf_auc:.4f}")

        # TIER 2 — SVM
        svm = SVC(
            kernel="rbf",
            C=svm_best[dataset_key][0],
            gamma=svm_best[dataset_key][1],
            class_weight="balanced",
            probability=True,
            random_state=RANDOM_SEED
        )
        
        # Naive epoch-level CV
        svm_ba_scores = []
        svm_auc_scores = []
        
        for train_idx, val_idx in skf.split(X, y):
            svm.fit(X[train_idx], y[train_idx])
            y_pred_proba = svm.predict_proba(X[val_idx])[:, 1]
            y_true = y[val_idx]
            
            svm_ba_scores.append(balanced_accuracy_score(y_true, y_pred_proba >= 0.5))
            svm_auc_scores.append(roc_auc_score(y_true, y_pred_proba))
        
        svm_ba = float(np.mean(svm_ba_scores))
        svm_auc = float(np.mean(svm_auc_scores))
        
        rows.append({
            "dataset": dataset_key,
            "tier": "tier2",
            "model": "svm",
            "balanced_accuracy": svm_ba,
            "roc_auc": svm_auc
        })
        print(f"  SVM: BA={svm_ba:.4f} AUC={svm_auc:.4f}")

        # TIER 3 — EEGNet  
        X_e, y_e, _ = build_dataset(dataset_key)
        
        # Naive epoch-level CV
        skf_eeg = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
        eegnet_ba_scores = []
        eegnet_auc_scores = []
        
        for train_idx, val_idx in skf_eeg.split(X_e, y_e):
            model = build_model(X_e.shape[1], X_e.shape[2], eg).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            
            # Build criterion with class weighting (same as in train_eegnet.py)
            if class_weight == "balanced":
                w = balanced_class_weights(y_e[train_idx])
                criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32).to(device))
            else:
                criterion = nn.CrossEntropyLoss()
            
            train_loader = DataLoader(EpochTensorDataset(X_e[train_idx], y_e[train_idx]), 
                                    batch_size=batch_size, shuffle=True)
            
            # Training loop for max_epochs
            for epoch in range(max_epochs):
                train_one_epoch(model, train_loader, optimizer, criterion, device)
            
            # Evaluation
            ba, auc = evaluate(model, X_e[val_idx], y_e[val_idx], device)
            eegnet_ba_scores.append(ba)
            eegnet_auc_scores.append(auc)
        
        eegnet_ba = float(np.mean(eegnet_ba_scores))
        eegnet_auc = float(np.mean(eegnet_auc_scores))
        
        rows.append({
            "dataset": dataset_key,
            "tier": "tier3",
            "model": "eegnet",
            "balanced_accuracy": eegnet_ba,
            "roc_auc": eegnet_auc
        })
        print(f"  EEGNet: BA={eegnet_ba:.4f} AUC={eegnet_auc:.4f}")

    # Save results to CSV
    results_df = pd.DataFrame(rows)
    results_df.to_csv(RESULTS_DIR / "tier23_naive_cv.csv", index=False)
    print(f"\nResults saved to {RESULTS_DIR / 'tier23_naive_cv.csv'}")

if __name__ == "__main__":
    main()