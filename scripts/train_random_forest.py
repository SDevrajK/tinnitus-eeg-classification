"""
Train Random Forest classifier on engineered feature matrix with subject-grouped CV configuration.

This script trains a Random Forest classifier using nested GroupKFold cross-validation
to prevent subject-level leakage, following the Tier 2 training pattern.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from _common import load_config, load_scaled_features, RESULTS_DIR, RANDOM_SEED

# Feature column helper
FEATURE_PREFIXES = ("power_", "wpli_", "plzc_")
def feature_columns(df):
    return [col for col in df.columns if any(col.startswith(prefix) for prefix in FEATURE_PREFIXES)]


def build_random_forest(config) -> RandomForestClassifier:
    """Build a Tier-2 RandomForestClassifier from config (the base model used by main's grid search)."""
    rf_cfg = config["tier2"]["random_forest"]
    return RandomForestClassifier(class_weight=rf_cfg["class_weight"], random_state=RANDOM_SEED)


def main():
    config = load_config()
    cv_folds = int(config["tier2"]["cv_folds"])
    scoring = config["tier2"]["scoring"]
    n_jobs = config["tier2"]["n_jobs"]
    rf_cfg = config["tier2"]["random_forest"]
    param_grid = {
        "n_estimators": list(rf_cfg["n_estimators_grid"]),
        "max_depth": list(rf_cfg["max_depth_grid"]),
        "min_samples_leaf": list(rf_cfg["min_samples_leaf_grid"]),
        "max_features": list(rf_cfg["max_features_grid"])
    }

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

        base = build_random_forest(config)

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
        joblib.dump(model, MODELS_DIR / f"{dataset_key}_random_forest.joblib")

        # Extract best params
        best_n_estimators = int(final_search.best_params_["n_estimators"])
        best_max_depth = final_search.best_params_["max_depth"]  # May be None
        best_min_samples_leaf = int(final_search.best_params_["min_samples_leaf"])
        best_max_features = final_search.best_params_["max_features"]  # May be float

        # Append row
        rows.append({
            "dataset": dataset_key,
            "best_n_estimators": best_n_estimators,
            "best_max_depth": best_max_depth,
            "best_min_samples_leaf": best_min_samples_leaf,
            "best_max_features": best_max_features,
            "nested_balanced_accuracy": nested_ba,
            "nested_roc_auc": nested_auc
        })

        # Print summary
        print(f"{dataset_key}: n_est={best_n_estimators}, depth={best_max_depth}, leaf={best_min_samples_leaf}, feat={best_max_features}, nested_BA={nested_ba:.4f}, nested_AUC={nested_auc:.4f}")

    # Save best parameters
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "random_forest_best_params.csv", index=False)

if __name__ == "__main__":
    main()