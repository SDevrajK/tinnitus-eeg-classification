"""
RBF-kernel SVM classifier for tier 2 analysis with nested GroupKFold CV.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from _common import load_config, load_scaled_features, RESULTS_DIR, RANDOM_SEED

# Feature column helper
FEATURE_PREFIXES = ("power_", "wpli_", "plzc_")
def feature_columns(df):
    return [col for col in df.columns if any(col.startswith(prefix) for prefix in FEATURE_PREFIXES)]

def main():
    config = load_config()
    cv_folds = int(config["tier2"]["cv_folds"])
    scoring = config["tier2"]["scoring"]
    n_jobs = config["tier2"]["n_jobs"]
    svm_cfg = config["tier2"]["svm"]
    param_grid = {"C": list(svm_cfg["C_grid"]), "gamma": list(svm_cfg["gamma_grid"])}
    class_weight = svm_cfg["class_weight"]

    MODELS_DIR = RESULTS_DIR / "models"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    dataset_keys = ("torres_torres", "ibarra_zarate", "raeisi", "wang")
    rows = []

    for dataset_key in dataset_keys:
        print(f"Training on {dataset_key}...")
        df = load_scaled_features(dataset_key)
        feat_cols = feature_columns(df)
        X = df[feat_cols].to_numpy()
        y = (df["Group"] == "Tinnitus").astype(int).to_numpy()
        groups = df["Subject_ID"].to_numpy()

        base = SVC(kernel="rbf", class_weight=class_weight, probability=True, random_state=RANDOM_SEED)

        # Nested CV
        outer = GroupKFold(n_splits=cv_folds)
        inner = GroupKFold(n_splits=cv_folds)
        
        fold_ba = []
        fold_auc = []
        
        for train_idx, val_idx in outer.split(X, y, groups=groups):
            gs = GridSearchCV(base, param_grid, cv=inner, scoring=scoring, n_jobs=n_jobs)
            gs.fit(X[train_idx], y[train_idx], groups=groups[train_idx])
            
            y_pred = gs.predict(X[val_idx])
            y_prob = gs.predict_proba(X[val_idx])[:, 1]
            
            fold_ba.append(balanced_accuracy_score(y[val_idx], y_pred))
            fold_auc.append(roc_auc_score(y[val_idx], y_prob))
        
        nested_ba = float(np.mean(fold_ba))
        nested_auc = float(np.mean(fold_auc))

        # Refit final model on full dataset
        final_search = GridSearchCV(base, param_grid, cv=GroupKFold(n_splits=cv_folds), scoring=scoring, n_jobs=n_jobs)
        final_search.fit(X, y, groups=groups)
        model = final_search.best_estimator_
        joblib.dump(model, MODELS_DIR / f"{dataset_key}_svm.joblib")

        # Extract best params
        best_C = float(final_search.best_params_["C"])
        best_gamma = final_search.best_params_["gamma"]  # May be string "scale"

        # Append row
        rows.append({
            "dataset": dataset_key,
            "best_C": best_C,
            "best_gamma": best_gamma,
            "nested_balanced_accuracy": nested_ba,
            "nested_roc_auc": nested_auc
        })

        # Print summary
        print(f"{dataset_key}: C={best_C}, gamma={best_gamma}, nested_BA={nested_ba:.4f}, nested_AUC={nested_auc:.4f}")

    # Save best parameters
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "svm_best_params.csv", index=False)

if __name__ == "__main__":
    main()