"""
Tier 2 (RF + RBF SVM) pairwise cross-dataset transfer (12 ordered pairs), reusing trained models.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from _common import load_scaled_features, RESULTS_DIR

DATASET_KEYS = ("torres_torres", "ibarra_zarate", "raeisi", "wang")
FEATURE_PREFIXES = ("power_", "wpli_", "plzc_")

def feature_columns(df):
    """Return column names that start with any of the FEATURE_PREFIXES."""
    return [col for col in df.columns if any(col.startswith(prefix) for prefix in FEATURE_PREFIXES)]

def main():
    MODELS_DIR = RESULTS_DIR / "models"
    rows = []
    
    # For each ordered pair of datasets
    for source in DATASET_KEYS:
        for target in DATASET_KEYS:
            if source != target:
                # Load BOTH source models once
                rf = joblib.load(MODELS_DIR / f"{source}_random_forest.joblib")
                svm = joblib.load(MODELS_DIR / f"{source}_svm.joblib")
                
                # Load target features
                df_t = load_scaled_features(target)
                feat_cols = feature_columns(df_t)
                X_t = df_t[feat_cols].to_numpy()
                y_t = (df_t["Group"] == "Tinnitus").astype(int).to_numpy()
                
                # Evaluate Random Forest
                y_prob = rf.predict_proba(X_t)[:, 1]
                y_pred = (y_prob >= 0.5).astype(int)
                ba = balanced_accuracy_score(y_t, y_pred)
                auc = roc_auc_score(y_t, y_prob)
                rows.append({
                    "source": source,
                    "target": target,
                    "tier": "tier2",
                    "model": "random_forest",
                    "balanced_accuracy": ba,
                    "roc_auc": auc
                })
                print(f"{source} -> {target}: RF BA={ba:.4f} AUC={auc:.4f}")
                
                # Evaluate RBF SVM
                y_prob = svm.predict_proba(X_t)[:, 1]
                y_pred = (y_prob >= 0.5).astype(int)
                ba = balanced_accuracy_score(y_t, y_pred)
                auc = roc_auc_score(y_t, y_prob)
                rows.append({
                    "source": source,
                    "target": target,
                    "tier": "tier2",
                    "model": "svm",
                    "balanced_accuracy": ba,
                    "roc_auc": auc
                })
                print(f"{source} -> {target}: SVM BA={ba:.4f} AUC={auc:.4f}")
    
    # Save results to CSV
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "pairwise_transfer_tier2.csv", index=False)

if __name__ == "__main__":
    main()