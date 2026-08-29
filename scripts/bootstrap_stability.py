"""
Bootstrap stability analysis for Tier 1 elastic-net models.

This script performs subject-level bootstrap resampling on the Tier 1 elastic-net
models per dataset to compute the selection frequency of each feature across
resamples. This follows the stability selection methodology from Meinshausen & Bühlmann (2010).

FR 8: Bootstrap stability analysis — compute selection frequency of coefficients
from Tier 1 models and visualize stability per feature.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.linear_model import SGDClassifier
from _common import load_config, load_scaled_features, RESULTS_DIR, FIGURES_DIR, RANDOM_SEED

DATASET_KEYS = ("torres_torres", "ibarra_zarate", "raeisi", "wang")
FEATURE_PREFIXES = ("power_", "wpli_", "plzc_")
RELATIVE_TOL = 0.05  # a feature is "selected" when |coef| > RELATIVE_TOL * max(|coef|) within that fit (relative threshold, robust to l1_ratio)


def feature_columns(df):
    """Extract feature column names from a DataFrame."""
    return [col for col in df.columns if col.startswith(FEATURE_PREFIXES)]


def make_model(alpha, l1_ratio, max_iter, tol):
    """Create a Tier 1 elastic-net model with fixed parameters."""
    # Same as the trained Tier 1 model with fixed random state
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
    n_bootstrap = int(config["phase4"]["n_bootstrap"])
    n_jobs = config["phase4"]["n_jobs"]
    max_iter = int(config["tier1"]["elastic_net"]["max_iter"])
    tol = float(config["tier1"]["elastic_net"]["tol"])

    # Load best hyperparameters
    best = pd.read_csv(RESULTS_DIR / "elastic_net_best_params.csv")
    best_map = {row["dataset"]: (row["best_alpha"], row["best_l1_ratio"]) for _, row in best.iterrows()}

    all_rows = []

    for dataset_key in DATASET_KEYS:
        print(f"Processing {dataset_key}...")
        # Load data
        df = load_scaled_features(dataset_key)
        feat_cols = feature_columns(df)
        X = df[feat_cols].to_numpy()
        y = (df["Group"] == "Tinnitus").astype(int).to_numpy()
        subjects = df["Subject_ID"].to_numpy()

        # Get model parameters for this dataset
        alpha, l1 = best_map[dataset_key]

        # SUBJECT-LEVEL BOOTSTRAP setup
        uniq = np.unique(subjects)
        rows_of = {s: np.where(subjects == s)[0] for s in uniq}
        
        # Precompute resamples
        rng = np.random.default_rng(RANDOM_SEED)
        resamples = [rng.choice(uniq, size=len(uniq), replace=True) for _ in range(n_bootstrap)]

        def _fit_one(sample_subjects):
            """Worker function for parallel resampling."""
            # Build the resampled row indices
            idx = np.concatenate([rows_of[s] for s in sample_subjects])
            X_sub = X[idx]
            y_sub = y[idx]
            
            # Train model with fixed hyperparameters
            m = make_model(alpha, l1, max_iter, tol)
            m.fit(X_sub, y_sub)
            
            # Get coefficients and return selection mask (relative threshold: robust for both lasso and elastic-net)
            coef = np.asarray(m.coef_).ravel()
            thr = RELATIVE_TOL * np.max(np.abs(coef))
            return (np.abs(coef) > thr).astype(np.int8)

        # Run the resamples in parallel
        masks = Parallel(n_jobs=n_jobs)(delayed(_fit_one)(rs) for rs in resamples)
        mask_stack = np.vstack(masks)  # shape: (n_bootstrap, 468)

        # Compute selection frequency
        selection_freq = mask_stack.mean(axis=0)

        # Store results
        for i, fname in enumerate(feat_cols):
            all_rows.append({
                "dataset": dataset_key,
                "feature": fname,
                "selection_frequency": float(selection_freq[i])
            })

        # Print summary for this dataset
        n_selected = int((selection_freq > 0).sum())
        n_stable = int((selection_freq > 0.8).sum())
        print(f"{dataset_key}: N_features_selected_ever={n_selected}, N_stable(>0.8)={n_stable}")

    # Save selection frequencies to CSV
    pd.DataFrame(all_rows).to_csv(RESULTS_DIR / "bootstrap_selection_frequency.csv", index=False)

    # Generate figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    # Color mapping for feature types
    color_map = {
        "power_": "blue",
        "wpli_": "green",
        "plzc_": "red"
    }

    for d, dataset_key in enumerate(DATASET_KEYS):
        # Reload selection frequencies for this dataset
        df_freq = pd.read_csv(RESULTS_DIR / "bootstrap_selection_frequency.csv")
        selection_freq = df_freq[df_freq["dataset"] == dataset_key]["selection_frequency"].values
        
        # Sort descending
        order = np.argsort(selection_freq)[::-1]
        selection_freq_sorted = selection_freq[order]
        
        # Create bar chart with color coding by feature type
        bars = axes[d].bar(np.arange(len(feat_cols)), selection_freq_sorted, width=1.0)
        
        # Color bars by feature type
        for i, (bar, fname) in enumerate(zip(bars, np.array(feat_cols)[order])):
            color = next((color_map[prefix] for prefix in color_map if fname.startswith(prefix)), "gray")
            bar.set_color(color)
        
        # Add threshold line
        axes[d].axhline(0.8, color="red", linestyle="--", linewidth=1)
        axes[d].set_ylim(0, 1)
        axes[d].set_xlabel("feature (sorted by selection frequency)")
        axes[d].set_ylabel("selection frequency")
        axes[d].set_title(dataset_key)

    fig.suptitle("Bootstrap coefficient selection frequency (subject-level, {} resamples)".format(n_bootstrap))
    fig.tight_layout()
    
    # Save figure
    figure_path = FIGURES_DIR / "tier1" / "bootstrap" / "bootstrap_selection_frequency.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()