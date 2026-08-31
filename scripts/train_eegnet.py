"""
Trains EEGNet-8,2 per dataset with subject-grouped CV; device-agnostic.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
from braindecode.models import EEGNet
from _common import load_config, RANDOM_SEED, RESULTS_DIR, FIGURES_DIR
from eegnet_data import build_dataset, EpochTensorDataset

def build_model(n_chans, n_times, eg):
    """Build EEGNet model with specified parameters."""
    return EEGNet(
        n_chans=n_chans,
        n_outputs=2,
        n_times=n_times,
        final_conv_length='auto',  # fixed braindecode idiom, not magic number
        pool_mode=eg["pool_mode"],
        F1=int(eg["F1"]),
        D=int(eg["D"]),
        F2=int(eg["F2"]),
        kernel_length=int(eg["kernel_length"]),
        drop_prob=float(eg["drop_prob"])
    )

def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    """Standard PyTorch training loop for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches

def evaluate(model, X_val, y_val, device) -> tuple[float, float]:
    """Evaluate model and return balanced accuracy and AUC."""
    model.eval()
    with torch.no_grad():
        X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val = torch.tensor(y_val, dtype=torch.long).to(device)
        logits = model(X_val)
        proba = torch.softmax(logits, dim=1)
        y_pred = torch.argmax(logits, dim=1)
        
        ba = balanced_accuracy_score(y_val.cpu(), y_pred.cpu())
        auc = roc_auc_score(y_val.cpu(), proba[:, 1].cpu())
        
    return ba, auc

def balanced_class_weights(y: np.ndarray) -> np.ndarray:
    """Return sklearn 'balanced' class weights for a binary label vector y (values 0/1)."""
    counts = np.bincount(y)
    # sklearn 'balanced' semantics: n_samples / (n_classes * count_per_class)
    return len(y) / (len(counts) * counts)

def main():
    config = load_config()
    t3 = config["tier3"]
    cv_folds = int(t3["cv_folds"])
    batch_size = int(t3["batch_size"])
    max_epochs = int(t3["max_epochs"])
    lr = float(t3["learning_rate"])
    weight_decay = float(t3["weight_decay"])
    class_weight = t3["class_weight"]
    eg = t3["eegnet"]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Set seeds for reproducibility
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if device == "cuda":
        torch.cuda.manual_seed_all(RANDOM_SEED)
    
    MODELS_DIR = RESULTS_DIR / "models"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    dataset_keys = ("torres_torres", "ibarra_zarate", "raeisi", "wang")
    cv_rows = []
    curve_data = []
    
    for dataset_key in dataset_keys:
        print(f"\nTraining on dataset: {dataset_key}")
        X, y, subject_ids = build_dataset(dataset_key)
        n_chans, n_times = X.shape[1], X.shape[2]
        
        gkf = GroupKFold(n_splits=cv_folds)
        fold_metrics = []
        fold_losses = []
        fold_val_bas = []
        
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=subject_ids)):
            print(f"  Fold {fold}")
            
            model = build_model(n_chans, n_times, eg).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            y_train_labels = y[train_idx]
            if class_weight == "balanced":
                w = balanced_class_weights(y_train_labels)
                criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32).to(device))
            else:
                criterion = nn.CrossEntropyLoss()
            
            train_ds = EpochTensorDataset(X[train_idx], y[train_idx])
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            
            epoch_losses = []
            epoch_val_bas = []
            
            for epoch in range(max_epochs):
                loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
                epoch_losses.append(loss)
                
                ba, _ = evaluate(model, X[val_idx], y[val_idx], device)
                epoch_val_bas.append(ba)
            
            # Compute final fold metrics
            ba, auc = evaluate(model, X[val_idx], y[val_idx], device)
            fold_metrics.append({"dataset": dataset_key, "fold": fold, "balanced_accuracy": ba, "roc_auc": auc})
            print(f"  {dataset_key} fold {fold}: BA={ba:.4f} AUC={auc:.4f}")
            
            # Store curve data for this fold
            fold_losses.append(epoch_losses)
            fold_val_bas.append(epoch_val_bas)
            
        # Average curves
        mean_train_losses = np.mean(fold_losses, axis=0)
        mean_val_bas = np.mean(fold_val_bas, axis=0)
        curve_data.append((dataset_key, mean_train_losses, mean_val_bas))
        
        # Final model training (full dataset)
        final_model = build_model(n_chans, n_times, eg).to(device)
        full_ds = EpochTensorDataset(X, y)
        full_loader = DataLoader(full_ds, batch_size=batch_size, shuffle=True)
        
        optimizer = optim.AdamW(final_model.parameters(), lr=lr, weight_decay=weight_decay)
        y_train_labels = y
        if class_weight == "balanced":
            w = balanced_class_weights(y_train_labels)
            criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32).to(device))
        else:
            criterion = nn.CrossEntropyLoss()
        
        for epoch in range(max_epochs):
            train_one_epoch(final_model, full_loader, optimizer, criterion, device)
        
        torch.save(final_model.state_dict(), MODELS_DIR / f"{dataset_key}_eegnet.pt")
        
        # Add fold metrics to cv_rows
        cv_rows.extend(fold_metrics)
    
    # Save cross-validation metrics
    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(RESULTS_DIR / "eegnet_cv_metrics.csv", index=False)
    
    # Plot training curves
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    dataset_titles = ["Dataset A", "Dataset B", "Dataset C", "Dataset D"]
    
    for i, (dataset_key, mean_train_losses, mean_val_bas) in enumerate(curve_data):
        ax1 = axes[i]
        ax2 = ax1.twinx()
        
        ax1.plot(range(1, max_epochs + 1), mean_train_losses, label='Train Loss', color='blue')
        ax2.plot(range(1, max_epochs + 1), mean_val_bas, label='Val Balanced Accuracy', color='red')
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Train Loss', color='blue')
        ax2.set_ylabel('Val Balanced Accuracy', color='red')
        ax1.set_title(dataset_titles[i])
        
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
    
    plt.tight_layout()
    
    # Save the figure
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    curves_fig_path = FIGURES_DIR / "tier3" / "eegnet_training_curves.png"
    curves_fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(curves_fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    main()