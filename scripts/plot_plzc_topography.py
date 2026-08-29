"""<docstring: single broadband scalp topography of PLZC group-difference effect size (Cohen's d), FDR-significant channels highlighted.>"""

import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
import pingouin as pg
from _common import load_config, load_features, load_shared_channels, FIGURES_DIR, RESULTS_DIR


def main() -> None:
    config = load_config()
    channels = load_shared_channels()
    n_ch = len(channels)
    fdr_results = pd.read_csv(RESULTS_DIR / "permutation_ttest.csv")

    for dataset_key in ("torres_torres", "ibarra_zarate", "raeisi", "wang"):
        df = load_features(dataset_key)
        feature_cols = [c for c in df.columns if c.startswith("plzc_")]
        assert len(feature_cols) == 13, f"Expected 13 plzc features, got {len(feature_cols)}"

        subject_means = df.groupby("Subject_ID", sort=True)[feature_cols].mean()
        subject_groups = df.drop_duplicates("Subject_ID").set_index("Subject_ID")["Group"]
        tinnitus = subject_means.loc[subject_groups == "Tinnitus"]
        control = subject_means.loc[subject_groups == "Control"]

        fdr_sub = fdr_results[fdr_results["dataset"] == dataset_key]

        effect = np.full(n_ch, np.nan)
        mask = np.zeros(n_ch, dtype=bool)
        for c, ch in enumerate(channels):
            fname = f"plzc_{ch}"
            if len(tinnitus[fname].dropna()) >= 2 and len(control[fname].dropna()) >= 2:
                effect[c] = pg.compute_effsize(tinnitus[fname].values, control[fname].values, paired=False, eftype="cohen")
            row = fdr_sub[fdr_sub["feature"] == fname]
            assert len(row) == 1, f"feature {fname} not found uniquely"
            mask[c] = bool(row["fdr_significant"].iloc[0])

        info = mne.create_info(ch_names=list(channels), sfreq=256, ch_types="eeg")
        info.set_montage("standard_1020")

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        vmax = np.nanmax(np.abs(effect))
        im, _ = mne.viz.plot_topomap(
            effect, info, axes=ax, cmap="RdBu_r", vlim=(-vmax, vmax),
            mask=mask, mask_params=dict(marker="o", markerfacecolor="w", markeredgecolor="k", markersize=7),
            sensors=True, names=list(channels), show=False, contours=0,
        )
        fig.colorbar(im, ax=ax, shrink=0.8, label="Cohen's d (Tinnitus - Control)")
        ax.set_title("PLZC")
        fig.suptitle(dataset_key)

        outdir = FIGURES_DIR / "tier0" / "permutation-t-test"
        outdir.mkdir(parents=True, exist_ok=True)
        path = outdir / f"{dataset_key}_plzc_topography.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"{dataset_key}: N_significant_channels={int(mask.sum())} -> {path}")


if __name__ == "__main__":
    main()