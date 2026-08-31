"""
4-class Random Forest dataset-of-origin confound check (subject-grouped CV).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from _common import load_config, load_scaled_features, RESULTS_DIR, RANDOM_SEED

DATASET_KEYS = ("torres_torres", "ibarra_zarate", "raeisi", "wang")

def feature_columns(df):
    FEATURE_PREFIXES = ("power_", "wpli_", "plzc_")
    return [col for col in df.columns if any(col.startswith(prefix) for prefix in FEATURE_PREFIXES)]

def main():
    config = load_config()
    cv_folds = int(config["tier2"]["cv_folds"])
    rf_cfg = config["tier2"]["lodo"]["random_forest"]
    
    # Pool all 4 datasets
    parts = []
    for dataset_key in DATASET_KEYS:
        df = load_scaled_features(dataset_key)
        df["dataset_label"] = dataset_key
        parts.append(df)
    pool = pd.concat(parts, ignore_index=True)
    
    X = pool[feature_columns(pool)].to_numpy()
    y = pool["dataset_label"].to_numpy()
    groups = pool["Subject_ID"].to_numpy()
    
    # Subject-grouped CV
    gkf = GroupKFold(n_splits=cv_folds)
    y_true_list = []
    y_pred_list = []
    
    for tr, va in gkf.split(X, y, groups=groups):
        rf = RandomForestClassifier(
            n_estimators=int(rf_cfg["n_estimators"]),
            max_depth=rf_cfg["max_depth"],
            min_samples_leaf=int(rf_cfg["min_samples_leaf"]),
            max_features=rf_cfg["max_features"],
            class_weight="balanced",
            random_state=RANDOM_SEED
        )
        rf.fit(X[tr], y[tr])
        y_true_list.append(y[va])
        y_pred_list.append(rf.predict(X[va]))
    
    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)
    
    # Compute metrics
    ba = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=DATASET_KEYS)
    
    # Save results
    pd.DataFrame({"balanced_accuracy": [ba]}).to_csv(RESULTS_DIR / "dataset_of_origin_rf_balanced_accuracy.csv", index=False)
    pd.DataFrame(cm, index=DATASET_KEYS, columns=DATASET_KEYS).to_csv(RESULTS_DIR / "dataset_of_origin_rf_confusion_matrix.csv")
    
    # Print results
    print(f"Balanced Accuracy: {ba}")
    print("Confusion Matrix:")
    print(cm)

if __name__ == "__main__":
    main()