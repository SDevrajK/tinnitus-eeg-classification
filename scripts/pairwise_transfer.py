"""
Pairwise cross-dataset transfer evaluation for Tier 1 elastic-net model.
Evaluates model trained on one dataset and tested on another, reporting
balanced accuracy and AUC-ROC for all 12 ordered pairs.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from _common import load_config, load_scaled_features, RESULTS_DIR, RANDOM_SEED

DATASET_KEYS = ("torres_torres", "ibarra_zarate", "raeisi", "wang")
FEATURE_PREFIXES = ("power_", "wpli_", "plzc_")

def feature_columns(df):
    """Return column names that start with any of the FEATURE_PREFIXES."""
    return [col for col in df.columns if any(col.startswith(prefix) for prefix in FEATURE_PREFIXES)]

def make_model(alpha, l1_ratio, max_iter, tol):
    """Create and return an SGDClassifier with elastic net penalty."""
    return SGDClassifier(
        loss="log_loss",
        penalty="elasticnet",
        alpha=alpha,
        l1_ratio=l1_ratio,
        max_iter=max_iter,
        tol=tol,
        random_state=RANDOM_SEED
    )

def main():
    config = load_config()
    max_iter = int(config["tier1"]["elastic_net"]["max_iter"])
    tol = float(config["tier1"]["elastic_net"]["tol"])
    
    # Load best hyperparameters
    best = pd.read_csv(RESULTS_DIR / "elastic_net_best_params.csv")
    best_map = {row["dataset"]: (row["best_alpha"], row["best_l1_ratio"]) for _, row in best.iterrows()}
    
    rows = []
    
    # For each ordered pair of datasets
    for source in DATASET_KEYS:
        for target in DATASET_KEYS:
            if source != target:
                # Load source data
                df_src = load_scaled_features(source)
                X_src = df_src[feature_columns(df_src)].to_numpy()
                y_src = (df_src["Group"] == "Tinnitus").astype(int).to_numpy()
                
                # Load target data
                df_tgt = load_scaled_features(target)
                X_tgt = df_tgt[feature_columns(df_tgt)].to_numpy()
                y_tgt = (df_tgt["Group"] == "Tinnitus").astype(int).to_numpy()
                
                # Get best hyperparameters for the source dataset
                alpha, l1 = best_map[source]
                model = make_model(alpha, l1, max_iter, tol)
                
                # Train on source, test on target
                model.fit(X_src, y_src)
                y_prob = model.predict_proba(X_tgt)[:, 1]
                
                # Calculate metrics
                ba = balanced_accuracy_score(y_tgt, (y_prob >= 0.5).astype(int))
                auc = roc_auc_score(y_tgt, y_prob)
                
                # Store results
                rows.append({
                    "source": source,
                    "target": target,
                    "balanced_accuracy": ba,
                    "roc_auc": auc
                })
                print(f"{source} -> {target}: BA={ba:.4f} AUC={auc:.4f}")
    
    # Save results to CSV
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "pairwise_transfer.csv", index=False)

if __name__ == "__main__":
    main()