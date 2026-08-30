"""
Computes Gini importance (Random Forest) and permutation importance (RBF SVM)
for the trained Tier 2 models, for global interpretability.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.inspection import permutation_importance

from _common import load_config, load_scaled_features, RESULTS_DIR, RANDOM_SEED

FEATURE_PREFIXES = ("power_", "wpli_", "plzc_")

def feature_columns(df):
    return [col for col in df.columns if any(col.startswith(prefix) for prefix in FEATURE_PREFIXES)]

def main():
    config = load_config()
    imp_cfg = config["tier2"]["importance"]
    n_repeats = int(imp_cfg["n_repeats"])
    scoring = imp_cfg["scoring"]
    n_jobs = config["tier2"]["n_jobs"]
    
    MODELS_DIR = RESULTS_DIR / "models"
    
    rows = []
    
    for dataset_key in ("torres_torres", "ibarra_zarate", "raeisi", "wang"):
        df = load_scaled_features(dataset_key)
        feat_cols = feature_columns(df)
        X = df[feat_cols].to_numpy()
        y = (df["Group"] == "Tinnitus").astype(int).to_numpy()
        
        # Load trained models
        rf = joblib.load(MODELS_DIR / f"{dataset_key}_random_forest.joblib")
        svm = joblib.load(MODELS_DIR / f"{dataset_key}_svm.joblib")
        
        # Gini (RF)
        rf_imp = rf.feature_importances_
        assert len(rf_imp) == len(feat_cols), f"RF importance length mismatch: {len(rf_imp)} vs {len(feat_cols)}"
        
        # Permutation (SVM)
        perm = permutation_importance(svm, X, y, scoring=scoring, n_repeats=n_repeats, random_state=RANDOM_SEED, n_jobs=n_jobs)
        svm_imp = perm.importances_mean
        assert len(svm_imp) == len(feat_cols), f"SVM importance length mismatch: {len(svm_imp)} vs {len(feat_cols)}"
        
        # Collect results
        for i in range(len(feat_cols)):
            rows.append({
                "dataset": dataset_key,
                "model": "random_forest",
                "feature": feat_cols[i],
                "importance": float(rf_imp[i])
            })
            rows.append({
                "dataset": dataset_key,
                "model": "svm",
                "feature": feat_cols[i],
                "importance": float(svm_imp[i])
            })
        
        print(f"{dataset_key}: RF Gini done; SVM permutation done (top feature = {feat_cols[int(np.argmax(svm_imp))]})")
    
    # Save results
    results_df = pd.DataFrame(rows)
    results_df.to_csv(RESULTS_DIR / "tier2_feature_importance.csv", index=False)

if __name__ == "__main__":
    main()