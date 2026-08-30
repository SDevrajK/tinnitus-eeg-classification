# Tinnitus EEG Classification — Interim README

> Interim project-wide README. The full, narrative README (research question,
> datasets, licensing, four-tier comparison, and findings) is a Phase 5
> deliverable. This file currently documents the end-to-end pipeline run order
> only.

## Environment

```bash
conda activate tinnitus-eeg
```

All pipeline scripts import `_common` directly and must be run with `scripts/`
as the working directory:

```bash
cd /home/sdevrajk/projects/personal/MachineLearning/scripts
```

## Pipeline run order

Run the scripts in this order to reproduce the full pipeline from raw data
through the Tier 0/1 baselines (the Phase 5 entry point).

### Phase 2 — Preprocessing

| # | Script | Purpose |
|---|--------|---------|
| 1 | `python preprocess_raw.py` | Load/standardize Datasets A/B/C (GDF/SET/NPZ), trim raeisi digitizer-startup artifact, bandpass + notch filter, bad-channel interpolation, epoch + artifact-reject, save `.fif`. Excludes montage-invalid and duplicate-recording subjects (from `config.yaml`). |

`preprocess_wang.py` is Dataset D only (already-preprocessed; skip filtering) and
does not need re-running for raeisi changes.

### Phase 3 — Feature extraction

| # | Script | Purpose |
|---|--------|---------|
| 2 | `python build_feature_matrix.py` | Assemble band-power + wPLI + PLZC features per epoch into `data/features/{dataset}_features.parquet`. Rebuilds all four datasets (no per-dataset flag). |

### Phase 4 — Tier 0 (statistical baseline)

| # | Script | Purpose |
|---|--------|---------|
| 3 | `python permutation_ttest.py` | Permutation t-tests + Benjamini–Hochberg FDR → `results/permutation_ttest.csv`. |
| 4 | `python plot_topography.py` | Band-power effect-size topomaps (FDR-masked) → `figures/tier0/permutation-t-test/`. |
| 5 | `python plot_wpli_heatmap.py` | wPLI connectivity heatmaps (FDR-masked) → `figures/tier0/permutation-t-test/`. |
| 6 | `python plot_plzc_topography.py` | PLZC broadband topomaps → `figures/tier0/permutation-t-test/`. |

### Phase 4 — Tier 1 (sparse linear model)

| # | Script | Purpose |
|---|--------|---------|
| 7 | `python robust_scaling.py` | Robust-scale features per dataset → `data/features/scaled/{dataset}_scaled.parquet`. |
| 8 | `python train_elastic_net.py` | Elastic-net logistic regression with nested CV → `results/models/{dataset}_elastic_net.joblib` + best params. |
| 9 | `python extract_coefficients.py` | Map fitted coefficients to channel/band/feature-type → `results/elastic_net_coefficients.csv`. |
| 10 | `python within_dataset_cv.py` | Naive vs. subject-grouped (LOSO) CV → `results/within_dataset_cv_comparison.csv` + figure. |
| 11 | `python pairwise_transfer.py` | Train-on-one/test-on-another (12 pairs) → `results/pairwise_transfer.csv`. |
| 12 | `python lodo_transfer.py` | Leave-one-dataset-out (4 configs) with epoch capping + class weighting → `results/lodo_transfer.csv`. |
| 13 | `python dataset_of_origin.py` | 4-class dataset-of-origin confound check → confusion matrix + balanced accuracy. |
| 14 | `python bootstrap_stability.py` | Bootstrap coefficient-selection frequency (≥1000 resamples) → `results/bootstrap_selection_frequency.csv` + figure. |
| 15 | `python cross_dataset_figure.py` | Pairwise + LODO summary figure → `figures/tier1/LODO/`. |

## Not run in this order

These scripts are acquisition/organization/one-time utilities and do **not**
need re-running as part of the analysis pipeline:

- `organize_bids.py`, `generate_inventory.py`, `compute_shared_channels.py`,
  `preprocess_wang.py` (Dataset D), `exclude_subjects.py`, `generate_qc_figures.py`.

## Recent data corrections (raeisi)

- **De-duplication:** `config.yaml` → `excluded_subjects_duplicates.raeisi`
  drops T11–T20 (float32 re-saves of T1–T10) and H14 (byte-identical to H13),
  so the analytic raeisi sample is 15 control / 10 tinnitus (25 unique), not
  16/20.
- **Onset trimming:** `preprocess_raw.py` → `_load_raeisi` trims the leading
  zero samples plus the single digitizer-startup spike (configurable via
  `preprocessing.raeisi_trim_threshold`) before resampling/filtering.
