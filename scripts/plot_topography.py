"""
Scalp topography figures visualizing group-difference effect size per frequency band.

Visualizes Cohen's d effect sizes for group differences (Tinnitus vs Control)
with FDR-significant channels highlighted. Uses MNE for plotting and pingouin
for computing effect sizes, following the PRD rationale to emphasize
statistically significant findings.

Based on Cohen (1988), effect size interpretation:
- Small: 0.2
- Medium: 0.5
- Large: 0.8
"""

import numpy as np
import pandas as pd
import mne
import pingouin as pg
from _common import load_config, load_features, load_shared_channels, FIGURES_DIR, RESULTS_DIR

def main():
    config = load_config()
    band_names = list(config["frequency_bands"].keys())
    channels = load_shared_channels()
    
    # Load FDR results
    fdr_results = pd.read_csv(RESULTS_DIR / "permutation_ttest.csv")
    
    # Process each dataset
    for dataset_key in ("torres_torres", "ibarra_zarate", "raeisi", "wang"):
        print(f"Processing {dataset_key}...")
        
        # Load data
        df = load_features(dataset_key)
        feature_cols = [c for c in df.columns if c.startswith("power_")]  # Should be 65
        assert len(feature_cols) == 65
        
        # Calculate subject-level means
        subject_means = df.groupby("Subject_ID", sort=True)[feature_cols].mean()
        subject_groups = df.drop_duplicates("Subject_ID").set_index("Subject_ID")["Group"]
        
        tinnitus = subject_means.loc[subject_groups == "Tinnitus"]
        control = subject_means.loc[subject_groups == "Control"]
        
        # Compute effect sizes (Cohen's d)
        effect = np.full((len(band_names), len(channels)), np.nan)
        for b, band in enumerate(band_names):
            for c, channel in enumerate(channels):
                feature_name = f"power_{band}_{channel}"
                # Check if both groups have sufficient samples
                if len(tinnitus[feature_name].dropna()) >= 2 and len(control[feature_name].dropna()) >= 2:
                    effect[b, c] = pg.compute_effsize(
                        tinnitus[feature_name].values,
                        control[feature_name].values,
                        paired=False,
                        eftype="cohen"
                    )
        
        # Build FDR significance mask
        fdr_subset = fdr_results[fdr_results["dataset"] == dataset_key]
        mask = np.full((len(band_names), len(channels)), False)
        
        for b, band in enumerate(band_names):
            for c, channel in enumerate(channels):
                feature_name = f"power_{band}_{channel}"
                fdr_row = fdr_subset[fdr_subset["feature"] == feature_name]
                assert len(fdr_row) == 1, f"Feature {feature_name} not found or duplicated in FDR results"
                mask[b, c] = fdr_row["fdr_significant"].iloc[0]
        
        # MNE setup
        info = mne.create_info(ch_names=list(channels), sfreq=256, ch_types="eeg")
        info.set_montage("standard_1020")
        
        # Create figure with 5 subplots in a row
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
        
        # Shared symmetric color scale
        vmax = np.nanmax(np.abs(effect))
        vlim = (-vmax, vmax)
        
        # Plot each band
        for b, band in enumerate(band_names):
            im, _ = mne.viz.plot_topomap(
                effect[b], 
                info, 
                axes=axes[b], 
                cmap="RdBu_r", 
                vlim=vlim,
                mask=mask[b],
                mask_params=dict(marker="o", markerfacecolor="w", markeredgecolor="k", markersize=7),
                sensors=True, 
                names=list(channels), 
                show=False, 
                contours=0
            )
            axes[b].set_title(band)
        
        # Add shared colorbar and suptitle
        cbar = fig.colorbar(im, ax=axes[-1], shrink=0.8)
        cbar.set_label("Cohen's d (Tinnitus - Control)")
        fig.suptitle(dataset_key)
        
        # Save figure
        outdir = FIGURES_DIR / "tier0" / "permutation-t-test"
        outdir.mkdir(parents=True, exist_ok=True)
        path = outdir / f"{dataset_key}_topography.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        # Print summary
        n_significant = np.sum(mask)
        print(f"{dataset_key}: N_significant_channels={n_significant} -> {path}")

if __name__ == "__main__":
    main()