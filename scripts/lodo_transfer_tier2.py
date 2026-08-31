"""
Tier 2 (RF + RBF SVM) LODO transfer (4 configurations).

This script evaluates Random Forest and RBF SVM models in a Leave-One-Dataset-Out (LODO)
configuration using 4 held-out dataset configurations. It implements subject-stratified
epoch capping (max 90 epochs per subject) and class weighting as required by the PRD.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from _common import load_scaled_features, load_config, EPOCH_DROP_SEED, RANDOM_SEED, RESULTS_DIR
from lodo_pool import load_pool_scaled_features, cap_epochs_per_subject

DATASET_KEYS = ("torres_torres", "ibarra_zarate", "raeisi", "wang")
FEATURE_PREFIXES = ("power_", "wpli_", "plzc_")

def feature_columns(df):
    """Return list of feature columns matching the feature prefixes."""
    return [col for col in df.columns if any(col.startswith(prefix) for prefix in FEATURE_PREFIXES)]

def main():
    config = load_config()
    epoch_cap = int(config["phase4"]["epoch_cap"])  # 90 preserves ~100% of Datasets A/B while trimming the long tails of C/D (PRD FR13)
    rf_cfg = config["tier2"]["lodo"]["random_forest"]
    svm_cfg = config["tier2"]["lodo"]["svm"]
    
    rows = []
    for held_out in DATASET_KEYS:
        # Build the training pool as the concatenation of the OTHER 3 datasets
        X_pool, y_pool, subj_pool = load_pool_scaled_features(held_out)
        X_pool, y_pool, _ = cap_epochs_per_subject(X_pool, y_pool, subj_pool, epoch_cap, EPOCH_DROP_SEED)
        
        # Load the held-out test set
        df_t = load_scaled_features(held_out)
        X_test = df_t[feature_columns(df_t)].to_numpy()
        y_test = (df_t["Group"]=="Tinnitus").astype(int).to_numpy()
        
        # Train Random Forest
        rf = RandomForestClassifier(
            n_estimators=int(rf_cfg["n_estimators"]),
            max_depth=rf_cfg["max_depth"],
            min_samples_leaf=int(rf_cfg["min_samples_leaf"]),
            max_features=rf_cfg["max_features"],
            class_weight="balanced",
            random_state=RANDOM_SEED
        )
        rf.fit(X_pool, y_pool)
        ba = balanced_accuracy_score(y_test, rf.predict(X_test))
        auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:,1])
        rows.append({
            "held_out": held_out,
            "tier": "tier2",
            "model": "random_forest",
            "balanced_accuracy": ba,
            "roc_auc": auc
        })
        
        # Train RBF SVM
        svm = SVC(
            kernel="rbf",
            C=float(svm_cfg["C"]),
            gamma=svm_cfg["gamma"],
            class_weight="balanced",
            probability=True,
            random_state=RANDOM_SEED
        )
        svm.fit(X_pool, y_pool)
        ba = balanced_accuracy_score(y_test, svm.predict(X_test))
        auc = roc_auc_score(y_test, svm.predict_proba(X_test)[:,1])
        rows.append({
            "held_out": held_out,
            "tier": "tier2",
            "model": "svm",
            "balanced_accuracy": ba,
            "roc_auc": auc
        })
        
        print(f"{held_out}: RF BA={ba:.4f} AUC={auc:.4f} | SVM BA={ba:.4f} AUC={auc:.4f}")
    
    # Save results
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "lodo_transfer_tier2.csv", index=False)

if __name__ == "__main__":
    main()