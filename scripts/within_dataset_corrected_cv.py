"""
Corrected (subject-grouped) within-dataset CV for Tiers 2 and 3;
fixed hyperparameters for a clean comparison against the naive scheme.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from _common import load_scaled_features, RANDOM_SEED, RESULTS_DIR, load_config

def main():
    cv_folds = int(load_config()["tier2"]["cv_folds"])
    
    # Load best hyperparameters for Tier 2
    rf_params_df = pd.read_csv(RESULTS_DIR / "random_forest_best_params.csv")
    svm_params_df = pd.read_csv(RESULTS_DIR / "svm_best_params.csv")
    
    # Build parameter maps
    rf_params_map = {}
    for _, row in rf_params_df.iterrows():
        rf_params_map[row["dataset"]] = {
            "n_estimators": int(row["best_n_estimators"]),
            "max_depth": int(row["best_max_depth"]) if not pd.isna(row["best_max_depth"]) else None,
            "min_samples_leaf": int(row["best_min_samples_leaf"]),
            "max_features": row["best_max_features"] if not pd.isna(row["best_max_features"]) else None
        }
    
    svm_params_map = {}
    for _, row in svm_params_df.iterrows():
        svm_params_map[row["dataset"]] = {
            "C": float(row["best_C"]),
            "gamma": row["best_gamma"]  # May be "scale"
        }
    
    # Dataset keys in fixed order
    datasets = ("torres_torres", "ibarra_zarate", "raeisi", "wang")
    
    rows = []
    
    for dataset_key in datasets:
        print(f"Processing {dataset_key}...")
        
        # Load data
        df = load_scaled_features(dataset_key)
        feat_cols = [col for col in df.columns if col.startswith(("power_", "wpli_", "plzc_"))]
        X = df[feat_cols].to_numpy()
        y = (df["Group"] == "Tinnitus").astype(int).to_numpy()
        groups = df["Subject_ID"].to_numpy()
        
        # Tier 2 - Random Forest
        rf_params = rf_params_map[dataset_key]
        rf_model = RandomForestClassifier(
            n_estimators=rf_params["n_estimators"],
            max_depth=rf_params["max_depth"],
            min_samples_leaf=rf_params["min_samples_leaf"],
            max_features=rf_params["max_features"],
            class_weight="balanced",
            random_state=RANDOM_SEED
        )
        
        rf_scores = []
        gkf = GroupKFold(n_splits=cv_folds)
        
        for train_idx, val_idx in gkf.split(X, y, groups=groups):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            rf_model.fit(X_train, y_train)
            y_pred_proba = rf_model.predict_proba(X_val)[:, 1]
            y_pred = rf_model.predict(X_val)
            
            ba = balanced_accuracy_score(y_val, y_pred)
            auc = roc_auc_score(y_val, y_pred_proba)
            
            rf_scores.append((ba, auc))
        
        rf_ba, rf_auc = np.mean(rf_scores, axis=0)
        
        # Tier 2 - SVM
        svm_params = svm_params_map[dataset_key]
        svm_model = SVC(
            kernel="rbf",
            C=svm_params["C"],
            gamma=svm_params["gamma"],
            class_weight="balanced",
            probability=True,
            random_state=RANDOM_SEED
        )
        
        svm_scores = []
        
        for train_idx, val_idx in gkf.split(X, y, groups=groups):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            svm_model.fit(X_train, y_train)
            y_pred_proba = svm_model.predict_proba(X_val)[:, 1]
            y_pred = svm_model.predict(X_val)
            
            ba = balanced_accuracy_score(y_val, y_pred)
            auc = roc_auc_score(y_val, y_pred_proba)
            
            svm_scores.append((ba, auc))
        
        svm_ba, svm_auc = np.mean(svm_scores, axis=0)
        
        # Tier 3 - EEGNet (already computed, subject-grouped)
        eegnet_df = pd.read_csv(RESULTS_DIR / "eegnet_cv_metrics.csv")
        df_t3 = eegnet_df[eegnet_df["dataset"] == dataset_key]
        
        ba = float(df_t3["balanced_accuracy"].mean())
        auc = float(df_t3["roc_auc"].mean())
        
        # Append results
        rows.append({
            "dataset": dataset_key,
            "tier": "tier2",
            "model": "random_forest",
            "balanced_accuracy": rf_ba,
            "roc_auc": rf_auc
        })
        
        rows.append({
            "dataset": dataset_key,
            "tier": "tier2",
            "model": "svm",
            "balanced_accuracy": svm_ba,
            "roc_auc": svm_auc
        })
        
        rows.append({
            "dataset": dataset_key,
            "tier": "tier3",
            "model": "eegnet",
            "balanced_accuracy": ba,
            "roc_auc": auc
        })
        
        # Print summary
        print(f"  RF: BA={rf_ba:.4f}, AUC={rf_auc:.4f}")
        print(f"  SVM: BA={svm_ba:.4f}, AUC={svm_auc:.4f}")
        print(f"  EEGNet: BA={ba:.4f}, AUC={auc:.4f}")
    
    # Write results
    results_df = pd.DataFrame(rows)
    results_df.to_csv(RESULTS_DIR / "tier23_corrected_cv.csv", index=False)
    
    print("Corrected CV results saved to tier23_corrected_cv.csv")

if __name__ == "__main__":
    main()