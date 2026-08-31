# Tinnitus EEG Classification: Power/Interpretability Tradeoff & Cross-Dataset Generalization

A reproducible machine-learning pipeline that classifies tinnitus vs. healthy controls from resting-state EEG across four public datasets, and that explicitly tests three methodological validity questions: subject-level leakage, cross-dataset generalization, and the interpretability-vs-complexity tradeoff.

## Summary

This project builds a four-tier model comparison — statistical baseline, sparse linear, classical nonlinear, and deep learning — and evaluates every predictive tier with the same rigorous validation schemes. The headline findings are sobering and reproducible: (1) engineered-feature models reach *perfect* accuracy under naive (leaky) cross-validation and collapse toward chance once subjects are separated between train and test; (2) no tier generalizes across independently collected datasets (pairwise and leave-one-dataset-out transfer are both near chance); and (3) increased model complexity does **not** buy more generalizable interpretability — the models do not agree on which channels matter, either across model families or across recording sites.

## Motivation

Published tinnitus-EEG classifiers report high accuracy (a 2026 systematic review and meta-analysis reports a pooled ~86.2% accuracy, AUC 0.878). But most of this literature uses small samples (often *n* < 40–100) and cross-validation schemes that do **not** separate subjects between the train and test partitions. Epochs from the same subject share subject-specific signal — most strikingly, the phase-lag-index (wPLI) connectivity features are literally identical across a subject's epochs — so a model that sees a subject's epochs in both train and test can partly learn *subject identity* rather than the tinnitus/control class, inflating the score.

Multiple recent papers and reviews (Liu et al. 2021; Ramezani & Bolhasani 2023; a 2026 arXiv cross-dataset study) explicitly flag this as a validity problem: reported accuracy tends to drop, sometimes toward chance, once subject-level leakage is removed, or once a model is evaluated on a second, independently collected dataset. The cross-dataset work further shows that when datasets differ in recording hardware and protocol, part of what looks like "generalizable" signal can instead be *dataset-specific* (site/platform) distributional shift rather than class-related signal.

This project therefore measures, explicitly and quantitatively: (a) the accuracy inflation attributable to subject-level leakage; (b) cross-dataset generalization via pairwise transfer and leave-one-dataset-out; (c) whether cross-dataset results could instead be explained by the datasets simply being technically distinguishable from one another; and (d) whether increasing model complexity buys generalizable insight, or merely fits dataset-specific noise.

## Datasets

Four independent public datasets, analyzed only in the resting-state, eyes-open condition:

- **Dataset A** — Torres-Torres et al., "Characterization of Tinnitus Through the Analysis of Electroencephalographic Activity," Mendeley Data, DOI 10.17632/fj7sskjdt7.5 (CC BY 4.0). 37 subjects (15 control / 22 tinnitus).
- **Dataset B** — Ibarra-Zárate et al., "Acoustic Therapies for Tinnitus Treatment: An EEG Database," Mendeley Data, DOI 10.17632/kj443jc4yc (CC BY 4.0). 103 subjects (14 control / 89 tinnitus); week-1 (pre-treatment) baseline session only.
- **Dataset C** — Raeisi, "EEG signal dataset in a resting state for individuals with tinnitus and healthy individuals," Zenodo, DOI 10.5281/zenodo.13308645 (CC BY 4.0). 36 subjects (16 control / 20 tinnitus).
- **Dataset D** — Wang et al., "Cross-Subject Tinnitus Diagnosis Based on Multi-Band EEG Contrastive Representation Learning," IEEE Journal of Biomedical and Health Informatics 2023, DOI 10.1109/JBHI.2023.3264521 (distributed via the authors' Google Drive). 267 acquired subjects (238 analytic after a 90 s minimum-data-length exclusion).

All cross-dataset analyses are restricted to the intersection of the **13 shared 10-20 channels** present in all four datasets: C3, C4, F3, F4, F7, F8, Fp1, Fp2, Fz, P7, P8, T7, T8.

## Methodology

**Four model tiers.** Tier 0 (statistical baseline): per-feature permutation *t*-tests with Benjamini–Hochberg FDR correction. Tier 1 (sparse linear): elastic-net-regularized logistic regression. Tier 2 (classical nonlinear): Random Forest and RBF-kernel Support Vector Machine. Tier 3 (deep learning): EEGNet-8,2 (Lawhern et al. 2018) trained directly on minimally processed epoch time series.

**Features (Tiers 0–2).** Per epoch and per shared channel: band power in five canonical bands (delta 1–4 Hz, theta 4–8 Hz, alpha 8–13 Hz, beta 13–30 Hz, gamma 30–45 Hz), weighted phase-lag index (wPLI) connectivity between every channel pair per band, and permutation Lempel-Ziv complexity (PLZC). Tier 3 ingests the raw (minimally processed) 2-second epochs directly.

**Validation.** Every predictive tier is evaluated three ways: (1) *within-dataset* — naive epoch-level cross-validation vs. corrected subject-grouped cross-validation, reporting balanced accuracy and AUC-ROC; (2) *pairwise cross-dataset transfer* — train on one dataset, test on another, for all 12 ordered pairs; and (3) *leave-one-dataset-out (LODO)* — train on the union of three datasets, test on the fully held-out fourth. Hyperparameter selection uses nested, subject-grouped cross-validation with fixed random seeds throughout, so the whole pipeline is reproducible.

## Pipeline run order

All scripts live in `scripts/` and are run with `scripts/` as the working directory under `conda activate tinnitus-eeg`.

| Phase | Scripts |
|-------|---------|
| Phase 3 — feature extraction | `build_feature_matrix.py` |
| Phase 4 — Tier 0 (statistical) | `permutation_ttest.py`, `plot_topography.py`, `plot_wpli_heatmap.py`, `plot_plzc_topography.py` |
| Phase 4 — Tier 1 (linear) | `robust_scaling.py`, `train_elastic_net.py`, `extract_coefficients.py`, `within_dataset_cv.py`, `pairwise_transfer.py`, `lodo_transfer.py`, `dataset_of_origin.py`, `bootstrap_stability.py`, `cross_dataset_figure.py` |
| Phase 5 — Tier 2 (nonlinear) | `train_random_forest.py`, `train_svm.py`, `within_dataset_naive_cv.py`, `within_dataset_corrected_cv.py`, `compute_tier2_importance.py`, `map_tier2_importance.py`, `plot_tier2_importance.py` |
| Phase 5 — Tier 3 (deep) | `train_eegnet.py`, `compute_eegnet_saliency.py`, `extract_eegnet_spatial_filters.py`, `plot_eegnet_interpretability.py` |
| Phase 5 — validation + confound | `pairwise_transfer_tier{1,2,3}.py`, `lodo_transfer_tier{2,3}.py`, `dataset_of_origin_rf.py` |
| Phase 5 — figures + synthesis | `plot_naive_vs_corrected.py`, `plot_pairwise_transfer.py`, `plot_lodo_transfer.py`, `plot_dataset_of_origin.py`, `synthesize_interpretability.py`, `plot_interpretability_heatmap.py`, `assess_interpretability_agreement.py` |

Acquisition/organization/one-time utilities (not part of the analysis pipeline): `organize_bids.py`, `generate_inventory.py`, `compute_shared_channels.py`, `preprocess_wang.py` (Dataset D), `exclude_subjects.py`, `generate_qc_figures.py`.

## Leakage: naive vs. corrected cross-validation

The single largest methodological issue in the tinnitus-EEG literature is subject-level leakage. We quantify it directly by running every tier under both a *naive* (leaky, epoch-level) and a *corrected* (subject-grouped) cross-validation, holding everything else fixed.

**Tiers 2 and 3** (balanced accuracy / AUC-ROC, naive → corrected):

| Dataset | Model | Naive BA | Naive AUC | Corrected BA | Corrected AUC |
|---------|-------|----------|-----------|--------------|---------------|
| A (Torres-Torres) | Random Forest | 1.00 | 1.00 | 0.62 | 0.69 |
| A | SVM | 0.97 | 1.00 | 0.63 | 0.66 |
| A | EEGNet | 0.78 | 0.93 | 0.52 | 0.58 |
| B (Ibarra-Zárate) | Random Forest | 1.00 | 1.00 | 0.50 | 0.57 |
| B | SVM | 0.96 | 1.00 | 0.62 | 0.72 |
| B | EEGNet | 0.61 | 0.91 | 0.51 | 0.61 |
| C (Raeisi) | Random Forest | 1.00 | 1.00 | 0.72 | 0.90 |
| C | SVM | 1.00 | 1.00 | 0.80 | 0.93 |
| C | EEGNet | 0.71 | 0.99 | 0.60 | 0.69 |
| D (Wang) | Random Forest | 1.00 | 1.00 | 0.55 | 0.80 |
| D | SVM | 0.99 | 1.00 | 0.62 | 0.74 |
| D | EEGNet | 0.57 | 0.78 | 0.50 | 0.64 |

**Tier 1** (elastic net, naive vs. leave-one-subject-out):

| Dataset | Naive BA | Naive AUC | Corrected BA | Corrected AUC |
|---------|----------|-----------|--------------|---------------|
| A | 0.96 | 0.99 | 0.63 | 0.60 |
| B | 0.99 | 0.99 | 0.51 | 0.52 |
| C | 1.00 | 1.00 | 0.80 | 0.76 |
| D | 0.85 | 0.97 | 0.59 | 0.67 |

![Naive vs corrected cross-validation, Tiers 2 & 3](figures/tier23/naive_vs_corrected.png)

![Naive vs corrected cross-validation, Tier 1](figures/tier1/LOSO/within_dataset_naive_vs_loso.png)

The pattern is stark and consistent: the engineered-feature models (Random Forest and SVM) reach **perfect (1.00) balanced accuracy** under naive CV and collapse toward chance under corrected CV — e.g. Random Forest falls from 1.00 to as low as 0.50 on Dataset B, a ~50-point inflation. EEGNet, trained on raw EEG, leaks less subject identity (the raw time series carries less subject-specific structure than the wPLI features), but it also extracts less signal — its corrected scores sit near 0.50–0.60. Only Dataset C (raeisi) is separable above chance once leakage is removed.

## Cross-dataset generalization: pairwise transfer & LODO

Leakage correction is necessary but not sufficient: a model can pass a corrected within-dataset test and still fail on new recording environments. We test this two ways.

**Pairwise transfer.** For each of the 12 ordered dataset pairs, a model trained on the entirety of the source dataset is tested on the entirety of the target dataset, for every tier. Balanced accuracy across all tiers and pairs ranges from ~0.32 to ~0.67, and AUC-ROC from ~0.41 to ~0.73 — i.e. essentially at chance for most pairs, and well below the corrected within-dataset scores.

![Pairwise cross-dataset transfer matrix](figures/tier23/pairwise_transfer_matrix.png)

**Leave-one-dataset-out (LODO).** For each of the 4 configurations, a model is trained on the union of the other three datasets (with subject-stratified epoch capping and class weighting) and tested on the fully held-out dataset. Balanced accuracy / AUC-ROC:

| Held out | Tier 1 (elastic net) | Tier 2 (RF) | Tier 2 (SVM) | Tier 3 (EEGNet) |
|----------|----------------------|-------------|--------------|-----------------|
| A | 0.56 / 0.72 | 0.53 / 0.70 | 0.49 / 0.65 | 0.53 / 0.46 |
| B | 0.58 / 0.60 | 0.49 / 0.54 | 0.64 / 0.66 | 0.50 / 0.45 |
| C | 0.54 / 0.50 | 0.50 / 0.29 | 0.47 / 0.40 | 0.50 / 0.61 |
| D | 0.50 / 0.61 | 0.51 / 0.67 | 0.55 / 0.58 | 0.50 / 0.50 |

![Leave-one-dataset-out transfer](figures/tier23/lodo_transfer.png)

Both schemes tell the same story: **no tier generalizes across datasets.** The apparent within-dataset signal is largely dataset-specific, not a stable neural marker of tinnitus. The next section rules out the alternative explanation — that this is simply because the datasets are trivially separable by hardware/site.

## Dataset-of-origin confound check

If the four datasets were trivially separable from their engineered features, then the cross-dataset transfer failures would be uninformative — the models could just be learning "which site," not "tinnitus vs. control." To test this, we trained a 4-class Random Forest to predict *which dataset* an epoch came from, using the same engineered features as the tinnitus classifiers, with subject-grouped cross-validation.

The dataset-of-origin classifier achieves a **balanced accuracy of 0.44** — only *moderately* above the 0.25 chance level for 4 classes. The confusion matrix shows the misclassification is largely one-directional into Dataset D: raeisi (C) is almost entirely misclassified as wang (D) (3,664/3,664 epochs), and a large share of ibarra (B) also lands in wang (7,310 epochs), while torres (A) and wang (D) are identified more reliably.

![Dataset-of-origin confusion matrix](figures/tier23/dataset_of_origin_confusion_matrix.png)

The implication for the transfer results is important: because the datasets are only *moderately* distinguishable (0.44, far from ceiling), the near-chance cross-dataset transfer reflects genuinely weak generalizable class signal, **not** a trivial site/hardware confound. If the datasets had been near-perfectly separable, the transfer failures would carry little information. As it stands, they are a meaningful — and sobering — measure of weak generalization. (The moderate separability does still imply *some* site-specific signal exists, so it is not entirely absent.)

## Interpretability synthesis & complexity tradeoff

Finally, we ask whether the models *agree* on which neural features matter. We aggregate each tier's interpretability to a common **channel level** (the 13 shared channels) — elastic-net coefficient magnitudes for Tier 1, RF/SVM importance for Tier 2, EEGNet spatial-filter magnitudes for Tier 3 — and rank-normalize each (tier, dataset) so the tiers are comparable.

![Cross-tier channel importance heatmap](figures/tier23/interpretability_heatmap.png)

Agreement, measured by Spearman rank correlation and top-5 channel overlap:

- **Cross-dataset agreement is low.** Spearman correlations are mostly near zero (roughly −0.42 to +0.63; means ~0.02–0.26 across tiers), and top-5 channel overlap is correspondingly low (Jaccard ~0.11–0.67). The channels flagged as important in one dataset are not reliably flagged in the others.
- **Cross-tier agreement is moderate and variable** (up to ~0.70 for RF/SVM ↔ EEGNet on Dataset A, but near zero on Dataset D). Different model families only partly agree even *within* a site.

### Conclusion: does complexity buy generalizable interpretability?

**No.** Increasing model complexity (Tier 3 vs. Tiers 0–1) does not yield more generalizable interpretability. The channels flagged as important in one dataset are not reliably flagged in the others, and the model families only partially agree even within a dataset. Combined with the near-chance cross-dataset generalization, the evidence is that the higher-complexity models — and, to a large extent, the simpler ones too — are fitting dataset-specific signal rather than discovering stable, generalizable neural markers of tinnitus. For this task, on this data, model complexity buys neither better generalization nor more trustworthy explanations.

## Reproducibility (Docker)

The pipeline runs end-to-end from **cleaned data** (the preprocessed `.fif` epochs and feature matrices already present); raw-data acquisition is a separate, partly manual step and is not part of the containerized run.

Build the image:

```bash
docker build -t tinnitus-eeg .
```

Run it with the data directory and the repository mounted:

```bash
docker run -it \
  -v /home/sdevrajk/media-hdd/researchdata/personal/MachineLearning/data:/home/sdevrajk/media-hdd/researchdata/personal/MachineLearning/data \
  -v "$(pwd)":/workspace \
  -w /workspace \
  tinnitus-eeg bash
```

Inside the container, activate the environment and run the full pipeline (Phase 3 → Phase 5):

```bash
source activate tinnitus-eeg && cd scripts && bash ../run_pipeline.sh
```

## Data corrections (Dataset C)

Dataset C (raeisi) required two corrections before analysis, both configurable in `config.yaml`:

- **De-duplication** — `excluded_subjects_duplicates.raeisi` drops T11–T20 (float32 re-saves of T1–T10) and H14 (byte-identical to H13), so the analytic raeisi sample is 15 control / 10 tinnitus (25 unique subjects).
- **Onset trimming** — `preprocess_raw.py` trims the leading zero samples and the single digitizer-startup spike (via `preprocessing.raeisi_trim_threshold`) before resampling/filtering.
