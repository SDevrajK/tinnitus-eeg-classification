"""
Aggregates per-tier interpretability to channel-level importance and rank-normalizes for the cross-tier heatmap.
"""

import numpy as np
import pandas as pd
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
from _common import RESULTS_DIR, load_shared_channels

def channel_importance_from_features(features_and_weights):
    """Given a list of (channel, weight) pairs where channel is a single name OR a "chA-chB" pair,
    accumulate a dict {channel_name: sum_of_abs_weights} over the 13 shared channels.
    For a pair "chA-chB", add the weight to BOTH chA and chB (each gets the full weight).
    """
    channel_weights = {ch: 0.0 for ch in load_shared_channels()}
    
    for channel, weight in features_and_weights:
        # Handle channel pairs (e.g., "T7-T8")
        if "-" in channel:
            ch_a, ch_b = channel.split("-")
            channel_weights[ch_a] += abs(weight)
            channel_weights[ch_b] += abs(weight)
        else:
            channel_weights[channel] += abs(weight)
    
    return channel_weights

def tier_channel_importance(tier, dataset):
    """Return a 13-vector of channel importances for a given tier and dataset."""
    shared_channels = load_shared_channels()
    
    if tier == "tier1":
        # Read elastic_net_coefficients.csv
        df = pd.read_csv(RESULTS_DIR / "elastic_net_coefficients.csv")
        df_filtered = df[df["dataset"] == dataset]
        features_and_weights = [(row["channel"], row["coefficient"]) for _, row in df_filtered.iterrows()]
        channel_weights = channel_importance_from_features(features_and_weights)
        return np.array([channel_weights[ch] for ch in shared_channels])
    
    elif tier == "tier2":
        # Read tier2_feature_importance.csv
        df = pd.read_csv(RESULTS_DIR / "tier2_feature_importance.csv")
        df_filtered = df[df["dataset"] == dataset]
        
        # Process each model
        model_importances = []
        for model in ["random_forest", "svm"]:
            df_model = df_filtered[df_filtered["model"] == model]
            
            # Parse features and collect importances
            features_and_importances = []
            for _, row in df_model.iterrows():
                feature_type, band, channel = parse_feature(row["feature"])
                features_and_importances.append((channel, row["importance"]))
            
            # Aggregate
            channel_weights = channel_importance_from_features(features_and_importances)
            importance_vector = np.array([channel_weights[ch] for ch in shared_channels])
            
            # Min-max normalize to [0, 1]
            min_val = importance_vector.min()
            max_val = importance_vector.max()
            if max_val == min_val:
                normalized = np.zeros_like(importance_vector)
            else:
                normalized = (importance_vector - min_val) / (max_val - min_val)
            model_importances.append(normalized)
        
        # Average the two normalized vectors
        return np.mean(model_importances, axis=0)
    
    elif tier == "tier3":
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for Tier 3 processing but is not available")
        # Load eegnet_spatial_filters.pt
        filters = torch.load(RESULTS_DIR / "eegnet_spatial_filters.pt", weights_only=False)
        w = filters[dataset]  # Shape: (16, 13)
        # Take mean across the first axis (average over filters)
        return np.abs(w).mean(axis=0)
    
    else:
        raise ValueError(f"Unknown tier: {tier}")

def parse_feature(feature_name):
    """Parse feature name into (feature_type, band, channel) using the same grammar as map_tier2_importance.py."""
    # power_<band>_<ch> → ch
    if feature_name.startswith('power_'):
        parts = feature_name.split('_')
        return 'power', parts[1], parts[2]
    # wpli_<band>_<chA>_<chB> → "chA-chB"
    elif feature_name.startswith('wpli_'):
        parts = feature_name.split('_')
        return 'wpli', parts[1], f"{parts[2]}-{parts[3]}"
    # plzc_<ch> → ch
    elif feature_name.startswith('plzc_'):
        parts = feature_name.split('_')
        return 'plzc', None, parts[1]
    else:
        raise ValueError(f"Unknown feature format: {feature_name}")

def main():
    """Main execution function."""
    shared_channels = load_shared_channels()
    datasets = ("torres_torres", "ibarra_zarate", "raeisi", "wang")
    tiers = ("tier1", "tier2", "tier3")
    
    rows = []
    
    # For every (tier, dataset) combination
    for tier in tiers:
        for dataset in datasets:
            # Compute the 13-vector of channel importances
            importance_vector = tier_channel_importance(tier, dataset)
            
            # Rank-normalize the vector to [0, 1] (per column)
            min_val = importance_vector.min()
            max_val = importance_vector.max()
            if max_val == min_val:
                normalized_vector = np.zeros_like(importance_vector)
            else:
                normalized_vector = (importance_vector - min_val) / (max_val - min_val)
            
            # Record results for each channel
            for i, channel in enumerate(shared_channels):
                rows.append({
                    "tier": tier,
                    "dataset": dataset,
                    "channel": channel,
                    "raw_importance": importance_vector[i],
                    "normalized_importance": normalized_vector[i]
                })
    
    # Write to CSV file
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "interpretability_channel_importance.csv", index=False)
    
    # Print summary per (tier, dataset)
    print("Channel importance summary per tier/dataset:")
    for tier in tiers:
        for dataset in datasets:
            subset = df[(df["tier"] == tier) & (df["dataset"] == dataset)]
            top_channels = subset.nlargest(3, "normalized_importance")["channel"].tolist()
            print(f"  {tier} / {dataset}: {top_channels}")

if __name__ == "__main__":
    main()