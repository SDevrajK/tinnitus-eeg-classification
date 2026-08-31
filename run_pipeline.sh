#!/usr/bin/env bash

# This script runs the full pipeline (Phases 3-5) end-to-end from CLEANED data
# It should be run from within the Docker container after activating the tinnitus-eeg environment
# Usage inside Docker container: source activate tinnitus-eeg && cd scripts && bash ../run_pipeline.sh

set -e

# Change to scripts directory
cd scripts

# Phase 3: Build feature matrix
echo "Running Phase 3: build_feature_matrix.py"
python build_feature_matrix.py

# Phase 4 (Tier 0)
echo "Running Phase 4 Tier 0: permutation_ttest.py"
python permutation_ttest.py
echo "Running Phase 4 Tier 0: plot_topography.py"
python plot_topography.py
echo "Running Phase 4 Tier 0: plot_wpli_heatmap.py"
python plot_wpli_heatmap.py
echo "Running Phase 4 Tier 0: plot_plzc_topography.py"
python plot_plzc_topography.py

# Phase 4 (Tier 1)
echo "Running Phase 4 Tier 1: robust_scaling.py"
python robust_scaling.py
echo "Running Phase 4 Tier 1: train_elastic_net.py"
python train_elastic_net.py
echo "Running Phase 4 Tier 1: extract_coefficients.py"
python extract_coefficients.py
echo "Running Phase 4 Tier 1: within_dataset_cv.py"
python within_dataset_cv.py
echo "Running Phase 4 Tier 1: pairwise_transfer.py"
python pairwise_transfer.py
echo "Running Phase 4 Tier 1: lodo_transfer.py"
python lodo_transfer.py
echo "Running Phase 4 Tier 1: dataset_of_origin.py"
python dataset_of_origin.py
echo "Running Phase 4 Tier 1: bootstrap_stability.py"
python bootstrap_stability.py
echo "Running Phase 4 Tier 1: cross_dataset_figure.py"
python cross_dataset_figure.py

# Phase 5 (Tier 2)
echo "Running Phase 5 Tier 2: train_random_forest.py"
python train_random_forest.py
echo "Running Phase 5 Tier 2: train_svm.py"
python train_svm.py
echo "Running Phase 5 Tier 2: within_dataset_naive_cv.py"
python within_dataset_naive_cv.py
echo "Running Phase 5 Tier 2: within_dataset_corrected_cv.py"
python within_dataset_corrected_cv.py
echo "Running Phase 5 Tier 2: compute_tier2_importance.py"
python compute_tier2_importance.py
echo "Running Phase 5 Tier 2: map_tier2_importance.py"
python map_tier2_importance.py
echo "Running Phase 5 Tier 2: plot_tier2_importance.py"
python plot_tier2_importance.py

# Phase 5 (Tier 3)
echo "Running Phase 5 Tier 3: train_eegnet.py"
python train_eegnet.py
echo "Running Phase 5 Tier 3: compute_eegnet_saliency.py"
python compute_eegnet_saliency.py
echo "Running Phase 5 Tier 3: extract_eegnet_spatial_filters.py"
python extract_eegnet_spatial_filters.py
echo "Running Phase 5 Tier 3: plot_eegnet_interpretability.py"
python plot_eegnet_interpretability.py

# Phase 5 (validation + confound)
echo "Running Phase 5 validation + confound: pairwise_transfer_tier1.py"
python pairwise_transfer_tier1.py
echo "Running Phase 5 validation + confound: pairwise_transfer_tier2.py"
python pairwise_transfer_tier2.py
echo "Running Phase 5 validation + confound: pairwise_transfer_tier3.py"
python pairwise_transfer_tier3.py
echo "Running Phase 5 validation + confound: lodo_transfer_tier2.py"
python lodo_transfer_tier2.py
echo "Running Phase 5 validation + confound: lodo_transfer_tier3.py"
python lodo_transfer_tier3.py
echo "Running Phase 5 validation + confound: dataset_of_origin_rf.py"
python dataset_of_origin_rf.py

# Phase 5 (figures + synthesis)
echo "Running Phase 5 figures + synthesis: plot_naive_vs_corrected.py"
python plot_naive_vs_corrected.py
echo "Running Phase 5 figures + synthesis: plot_pairwise_transfer.py"
python plot_pairwise_transfer.py
echo "Running Phase 5 figures + synthesis: plot_lodo_transfer.py"
python plot_lodo_transfer.py
echo "Running Phase 5 figures + synthesis: plot_dataset_of_origin.py"
python plot_dataset_of_origin.py
echo "Running Phase 5 figures + synthesis: synthesize_interpretability.py"
python synthesize_interpretability.py
echo "Running Phase 5 figures + synthesis: plot_interpretability_heatmap.py"
python plot_interpretability_heatmap.py
echo "Running Phase 5 figures + synthesis: assess_interpretability_agreement.py"
python assess_interpretability_agreement.py

echo "Pipeline completed successfully!"