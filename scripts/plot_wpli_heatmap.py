"""
Generate connectivity HEATMAP figures for the wPLI features.
Per dataset, one figure with 5 subplots (one per frequency band: delta, theta, alpha, beta, gamma),
each a 13×13 channel×channel heatmap of the group-difference effect size (Cohen's d) per connection,
with FDR-significant connections marked.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pingouin as pg
from _common import load_config, load_features, load_shared_channels, FIGURES_DIR, RESULTS_DIR

def main():
    config = load_config()
    band_names = list(config["frequency_bands"].keys())
    channels = load_shared_channels()
    n_ch = len(channels)
    
    # Read FDR results
    fdr_results = pd.read_csv(RESULTS_DIR / "permutation_ttest.csv")
    
    # Dataset keys in order
    dataset_keys = ("torres_torres", "ibarra_zarate", "raeisi", "wang")
    
    for dataset_key in dataset_keys:
        print(f"Processing {dataset_key}...")
        
        # Load features and compute subject-level means
        df = load_features(dataset_key)
        subject_means = df.groupby("Subject_ID", sort=True)[[c for c in df.columns if c.startswith("wpli_")]].mean()
        subject_groups = df.drop_duplicates("Subject_ID").set_index("Subject_ID")["Group"]
        tinnitus = subject_means.loc[subject_groups == "Tinnitus"]
        control = subject_means.loc[subject_groups == "Control"]
        
        # Filter FDR results for this dataset
        fdr_sub = fdr_results[fdr_results["dataset"] == dataset_key]
        
        # Build per-band 13×13 matrices of effect size and significance
        effect = np.full((len(band_names), n_ch, n_ch), np.nan)
        mask = np.zeros((len(band_names), n_ch, n_ch), dtype=bool)
        
        for b, band in enumerate(band_names):
            for a in range(n_ch):
                for b2 in range(a+1, n_ch):  # Only compute upper triangle
                    fname = f"wpli_{band}_{channels[a]}_{channels[b2]}"
                    
                    # Check if we have at least 2 subjects in each group
                    tinnitus_vals = tinnitus[fname].values
                    control_vals = control[fname].values
                    
                    if len(tinnitus_vals[~np.isnan(tinnitus_vals)]) >= 2 and len(control_vals[~np.isnan(control_vals)]) >= 2:
                        d = pg.compute_effsize(tinnitus_vals, control_vals, paired=False, eftype="cohen")
                        effect[b, a, b2] = effect[b, b2, a] = d
                        mask[b, a, b2] = mask[b, b2, a] = bool(fdr_sub[fdr_sub["feature"] == fname]["fdr_significant"].iloc[0])
                    else:
                        # If either group has less than 2 non-NaN subjects, use NaN for effect size
                        effect[b, a, b2] = effect[b, b2, a] = np.nan
                        mask[b, a, b2] = mask[b, b2, a] = False
        
        # Create figure
        fig, axes = plt.subplots(1, 5, figsize=(26, 5))
        vmax = np.nanmax(np.abs(effect))
        
        for b, band in enumerate(band_names):
            im = axes[b].imshow(effect[b], cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
            axes[b].set_xticks(np.arange(n_ch))
            axes[b].set_xticklabels(channels, rotation=90, fontsize=8)
            axes[b].set_yticks(np.arange(n_ch))
            axes[b].set_yticklabels(channels, fontsize=8)
            axes[b].set_title(band)
            
            # Overlay FDR-significance markers
            for a in range(n_ch):
                for b2 in range(n_ch):
                    if mask[b, a, b2]:
                        axes[b].scatter(b2, a, marker="o", s=45, facecolors="none", edgecolors="black", linewidths=1.2)
        
        # Add colorbar
        cmap = plt.get_cmap("RdBu_r")
        cmap.set_bad("lightgray")
        fig.colorbar(im, ax=axes, shrink=0.8)
        axes[0].set_ylabel("Channel")
        fig.suptitle(dataset_key)
        plt.tight_layout()
        
        # Save figure
        outdir = FIGURES_DIR / "tier0" / "permutation-t-test"
        outdir.mkdir(parents=True, exist_ok=True)
        path = outdir / f"{dataset_key}_wpli_heatmap.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        # Print summary
        n_significant = int(mask.sum())
        print(f"{dataset_key}: N_significant_connections={n_significant} -> {path}")

if __name__ == "__main__":
    main()