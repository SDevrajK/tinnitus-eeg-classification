"""
Cross-dataset and cross-tier interpretability agreement (Spearman rank correlation + top-5 channel overlap).
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from _common import RESULTS_DIR

def main():
    # Load the CSV data
    df = pd.read_csv(RESULTS_DIR / "interpretability_channel_importance.csv")
    
    # Define dataset keys and display names
    dataset_keys = ['torres_torres', 'ibarra_zarate', 'raeisi', 'wang']
    dataset_names = ['A', 'B', 'C', 'D']
    dataset_map = dict(zip(dataset_keys, dataset_names))
    
    # Define tier names
    tiers = ['tier1', 'tier2', 'tier3']
    
    # Helper function to get vector for a (tier, dataset) combination
    def vector(tier, dataset):
        subset = df[(df['tier'] == tier) & (df['dataset'] == dataset)]
        return subset['raw_importance'].values
    
    # Helper function to get top-5 channels for a (tier, dataset) combination
    def top5(tier, dataset):
        subset = df[(df['tier'] == tier) & (df['dataset'] == dataset)]
        # Sort by importance descending, then by channel name ascending for tie-breaking
        sorted_indices = np.argsort(-subset['raw_importance'].values) 
        top5_indices = sorted_indices[:5]
        top5_channels = subset.iloc[top5_indices]['channel'].tolist()
        return set(top5_channels)
    
    rows = []
    
    # Compute CROSS-DATASET agreement (within each tier)
    for tier in tiers:
        print(f"Computing cross-dataset agreement for {tier}")
        # Generate all unordered pairs (A-B, A-C, A-D, B-C, B-D, C-D) for 4 datasets
        pairs = []
        for i in range(len(dataset_keys)):
            for j in range(i+1, len(dataset_keys)):
                pairs.append((dataset_keys[i], dataset_keys[j]))
        
        for d1_key, d2_key in pairs:
            # Get vectors
            v1 = vector(tier, d1_key)
            v2 = vector(tier, d2_key)
            
            # Compute Spearman correlation
            spearman = spearmanr(v1, v2).statistic
            
            # Compute top-5 Jaccard similarity
            top5_d1 = top5(tier, d1_key)
            top5_d2 = top5(tier, d2_key)
            intersection = len(top5_d1 & top5_d2)
            union = len(top5_d1 | top5_d2)
            jaccard = intersection / union if union > 0 else 0.0
            
            rows.append({
                "comparison": "cross_dataset",
                "group": tier,
                "pair": f"{dataset_map[d1_key]}-{dataset_map[d2_key]}",
                "spearman_corr": spearman,
                "top5_jaccard": jaccard
            })
    
    # Compute CROSS-TIER agreement (within each dataset)
    for dataset in dataset_keys:
        print(f"Computing cross-tier agreement for {dataset}")
        # Generate all tier pairs (tier1-tier2, tier1-tier3, tier2-tier3)
        pairs = []
        for i in range(len(tiers)):
            for j in range(i+1, len(tiers)):
                pairs.append((tiers[i], tiers[j]))
        
        for t1, t2 in pairs:
            # Get vectors
            v1 = vector(t1, dataset)
            v2 = vector(t2, dataset)
            
            # Compute Spearman correlation
            spearman = spearmanr(v1, v2).statistic
            
            # Compute top-5 Jaccard similarity
            top5_t1 = top5(t1, dataset)
            top5_t2 = top5(t2, dataset)
            intersection = len(top5_t1 & top5_t2)
            union = len(top5_t1 | top5_t2)
            jaccard = intersection / union if union > 0 else 0.0
            
            rows.append({
                "comparison": "cross_tier",
                "group": dataset,
                "pair": f"{t1[4:]}-{t2[4:]}",  # Remove "tier" prefix for cleaner display
                "spearman_corr": spearman,
                "top5_jaccard": jaccard
            })
    
    # Create DataFrame and save
    result_df = pd.DataFrame(rows)
    result_df.to_csv(RESULTS_DIR / "interpretability_agreement.csv", index=False)
    
    # Print summary
    print("\n=== SUMMARY ===")
    
    # Cross-dataset: mean spearman per tier
    cross_dataset_means = result_df[result_df['comparison'] == 'cross_dataset'].groupby('group')['spearman_corr'].mean()
    print("Cross-dataset Spearman correlation (mean per tier):")
    for tier, mean_corr in cross_dataset_means.items():
        print(f"  {tier}: {mean_corr:.3f}")
    
    # Cross-tier: mean spearman per tier-pair
    cross_tier_means = result_df[result_df['comparison'] == 'cross_tier'].groupby('pair')['spearman_corr'].mean()
    print("Cross-tier Spearman correlation (mean per tier-pair):")
    for pair, mean_corr in cross_tier_means.items():
        print(f"  {pair}: {mean_corr:.3f}")

if __name__ == "__main__":
    main()