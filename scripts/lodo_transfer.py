"""
LODO Transfer Evaluation for Tier 1 Elastic-Net Model

This script evaluates the Tier 1 elastic-net model in a Leave-One-Dataset-Out (LODO)
configuration using 4 held-out dataset configurations. It implements subject-stratified
epoch capping (max 90 epochs per subject) and class weighting as required by the PRD.

FR 13: Subject-stratified capping to control for subject bias
FR 14: Class weighting to handle imbalanced datasets
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from _common import load_config, load_scaled_features, RESULTS_DIR, RANDOM_SEED, EPOCH_DROP_SEED

DATASET_KEYS = ("torres_torres", "ibarra_zarate", "raeisi", "wang")
FEATURE_PREFIXES = ("power_", "wpli_", "plzc_")

def feature_columns(df):
    """Return list of feature columns matching the feature prefixes."""
    return [col for col in df.columns if any(col.startswith(prefix) for prefix in FEATURE_PREFIXES)]

def cap_epochs_per_subject(df: pd.DataFrame, max_epochs: int, seed: int) -> pd.DataFrame:
    """
    Apply subject-stratified epoch capping to limit max_epochs per subject.
    
    FR 13: Subject-stratified capping to control for subject bias
    """
    rng = np.random.default_rng(seed)
    capped_dfs = []
    
    for subject_id, subject_df in df.groupby('Subject_ID'):
        if len(subject_df) > max_epochs:
            # Randomly sample exactly max_epochs rows for this subject
            sampled_indices = rng.choice(subject_df.index, size=max_epochs, replace=False)
            capped_dfs.append(subject_df.loc[sampled_indices])
        else:
            # Keep all rows for this subject
            capped_dfs.append(subject_df)
    
    # Concatenate all capped subjects back together
    return pd.concat(capped_dfs, ignore_index=True)

def make_model(alpha, l1_ratio, max_iter, tol):
    """
    Create an SGDClassifier with elastic-net regularization and balanced class weights.
    
    FR 14: Class weighting to handle imbalanced datasets
    """
    return SGDClassifier(
        loss="log_loss",
        penalty="elasticnet",
        alpha=alpha,
        l1_ratio=l1_ratio,
        class_weight="balanced",  # This implements class weighting requirement
        max_iter=max_iter,
        tol=tol,
        random_state=RANDOM_SEED
    )

def main():
    config = load_config()
    epoch_cap = int(config["phase4"]["epoch_cap"])
    max_iter = int(config["tier1"]["elastic_net"]["max_iter"])
    tol = float(config["tier1"]["elastic_net"]["tol"])
    
    # FIXED LODO hyperparameters (documented reference config)
    LODO_ALPHA = 0.001
    LODO_L1_RATIO = 0.5
    
    rows = []
    for held_out in DATASET_KEYS:
        # Build the training pool as the concatenation of the OTHER 3 datasets
        pool_parts = [load_scaled_features(k) for k in DATASET_KEYS if k != held_out]
        pool = pd.concat(pool_parts, ignore_index=True)
        
        # Apply capping: subject-stratified epoch capping (FR 13)
        pool = cap_epochs_per_subject(pool, epoch_cap, EPOCH_DROP_SEED)
        
        # Prepare training data
        X = pool[feature_columns(pool)].to_numpy()
        y = (pool["Group"] == "Tinnitus").astype(int).to_numpy()
        
        # Make and train model with class weighting (FR 14)
        model = make_model(LODO_ALPHA, LODO_L1_RATIO, max_iter, tol)
        model.fit(X, y)
        
        # Load held-out dataset
        df_t = load_scaled_features(held_out)
        X_t = df_t[feature_columns(df_t)].to_numpy()
        y_t = (df_t["Group"] == "Tinnitus").astype(int).to_numpy()
        
        # Predict and evaluate
        y_prob = model.predict_proba(X_t)[:, 1]
        ba = balanced_accuracy_score(y_t, (y_prob >= 0.5).astype(int))
        auc = roc_auc_score(y_t, y_prob)
        
        # Record results
        rows.append({
            "held_out": held_out,
            "balanced_accuracy": ba,
            "roc_auc": auc
        })
        
        print(f"held_out={held_out}: BA={ba:.4f} AUC={auc:.4f} (pool_epochs_after_cap={len(pool)})")
    
    # Save results
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "lodo_transfer.csv", index=False)

if __name__ == "__main__":
    main()