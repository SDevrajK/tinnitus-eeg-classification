"""
Evaluate Tier 1 elastic-net model within each dataset under naive vs. LOSO CV schemes.

This script compares cross-validation performance between naive epoch-level splitting
and corrected Leave-One-Subject-Out (LOSO) cross-validation methods. It uses already-
tuned hyperparameters from Task 3 and computes balanced accuracy and AUC-ROC metrics.
"""

import warnings
warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.base import clone
from joblib import Parallel, delayed

from _common import load_config, load_scaled_features, RESULTS_DIR, FIGURES_DIR, RANDOM_SEED

FEATURE_PREFIXES = ("power_", "wpli_", "plzc_")

def feature_columns(df):
    """Return list of feature column names based on FEATURE_PREFIXES."""
    return [col for col in df.columns if any(col.startswith(prefix) for prefix in FEATURE_PREFIXES)]

def make_model(alpha, l1_ratio, max_iter, tol):
    """Create SGDClassifier with elastic-net penalty."""
    return SGDClassifier(
        loss="log_loss", 
        penalty="elasticnet", 
        alpha=alpha, 
        l1_ratio=l1_ratio, 
        max_iter=max_iter, 
        tol=tol, 
        random_state=RANDOM_SEED
    )

def naive_cv(X, y, model, n_splits, seed, n_jobs):
    """Naive epoch-level CV (StratifiedKFold), parallelized across folds."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(skf.split(X, y))

    def _fit_predict(tr, va):
        m = clone(model)
        m.fit(X[tr], y[tr])
        return y[va], m.predict_proba(X[va])[:, 1]

    results = Parallel(n_jobs=n_jobs)(delayed(_fit_predict)(tr, va) for tr, va in splits)
    y_true = np.concatenate([r[0] for r in results])
    y_prob = np.concatenate([r[1] for r in results])
    return y_true, y_prob

def loso_cv(X, y, groups, model, n_jobs):
    """Corrected Leave-One-Subject-Out CV, parallelized across subjects."""
    logo = LeaveOneGroupOut()
    splits = list(logo.split(X, y, groups=groups))

    def _fit_predict(tr, va):
        m = clone(model)
        m.fit(X[tr], y[tr])
        return y[va], m.predict_proba(X[va])[:, 1]

    results = Parallel(n_jobs=n_jobs)(delayed(_fit_predict)(tr, va) for tr, va in splits)
    y_true = np.concatenate([r[0] for r in results])
    y_prob = np.concatenate([r[1] for r in results])
    return y_true, y_prob

def main():
    config = load_config()
    n_jobs = config["phase4"]["n_jobs"]
    n_splits = int(config["tier1"]["cv_folds"])
    max_iter = int(config["tier1"]["elastic_net"]["max_iter"])
    tol = float(config["tier1"]["elastic_net"]["tol"])
    
    # Load best hyperparameters from Task 3
    best = pd.read_csv(RESULTS_DIR / "elastic_net_best_params.csv")
    best_map = {row["dataset"]: (row["best_alpha"], row["best_l1_ratio"]) for _, row in best.iterrows()}
    
    rows = []
    
    # Process each dataset
    for dataset_key in ("torres_torres", "ibarra_zarate", "raeisi", "wang"):
        df = load_scaled_features(dataset_key)
        feat_cols = feature_columns(df)
        X = df[feat_cols].to_numpy()
        y = (df["Group"] == "Tinnitus").astype(int).to_numpy()
        groups = df["Subject_ID"].to_numpy()
        
        # Get best hyperparameters for this dataset
        alpha, l1 = best_map[dataset_key]
        model = make_model(alpha, l1, max_iter, tol)
        
        # Run both CV schemes
        yt_n, yp_n = naive_cv(X, y, model, n_splits, RANDOM_SEED, n_jobs)
        yt_l, yp_l = loso_cv(X, y, groups, model, n_jobs)
        
        # Compute metrics for both schemes
        ba_n = balanced_accuracy_score(yt_n, (yp_n >= 0.5).astype(int))
        auc_n = roc_auc_score(yt_n, yp_n)
        ba_l = balanced_accuracy_score(yt_l, (yp_l >= 0.5).astype(int))
        auc_l = roc_auc_score(yt_l, yp_l)
        
        # Save results
        rows.append({
            "dataset": dataset_key,
            "scheme": "naive",
            "balanced_accuracy": ba_n,
            "roc_auc": auc_n
        })
        rows.append({
            "dataset": dataset_key,
            "scheme": "loso",
            "balanced_accuracy": ba_l,
            "roc_auc": auc_l
        })
        
        print(f"{dataset_key}: naive BA={ba_n:.4f} AUC={auc_n:.4f} | LOSO BA={ba_l:.4f} AUC={auc_l:.4f}")
    
    # Save results to CSV
    results_df = pd.DataFrame(rows)
    results_df.to_csv(RESULTS_DIR / "within_dataset_cv_comparison.csv", index=False)
    
    # Create comparison figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for metric_idx, (metric, ax) in enumerate([("balanced_accuracy", axes[0]), ("roc_auc", axes[1])]):
        # Prepare data for grouped bar chart
        naive_vals = results_df[results_df["scheme"] == "naive"][metric].values
        loso_vals = results_df[results_df["scheme"] == "loso"][metric].values
        x = np.arange(4)
        
        # Create bar chart
        ax.bar(x - 0.2, naive_vals, 0.35, label="Naive", alpha=0.8)
        ax.bar(x + 0.2, loso_vals, 0.35, label="LOSO", alpha=0.8)
        
        # Format chart
        ax.set_xticks(x)
        ax.set_xticklabels(["torres_torres", "ibarra_zarate", "raeisi", "wang"], rotation=30)
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.legend()
    
    fig.suptitle("Within-dataset CV: naive (leaky) vs LOSO (corrected)")
    fig.tight_layout()
    
    # Save figure
    figure_path = FIGURES_DIR / "tier1" / "LOSO" / "within_dataset_naive_vs_loso.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    print(f"Results saved to {RESULTS_DIR / 'within_dataset_cv_comparison.csv'}")
    print(f"Figure saved to {figure_path}")

if __name__ == "__main__":
    main()