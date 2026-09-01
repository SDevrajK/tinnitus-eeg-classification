#!/usr/bin/env python3
"""
Generate cross-tier interpretability summary heatmap.
"""

import pandas as pd
import matplotlib.pyplot as plt
from _common import RESULTS_DIR, FIGURES_DIR

def main():
    # Read the CSV file
    csv_path = RESULTS_DIR / "interpretability_channel_importance.csv"
    df = pd.read_csv(csv_path)
    
    # Canonical channel order (from shared_channels.json)
    channels = ['C3', 'C4', 'F3', 'F4', 'F7', 'F8', 'Fp1', 'Fp2', 'Fz', 'P7', 'P8', 'T7', 'T8']
    
    # Canonical channel order and dataset keys/display labels
    channels = ['C3', 'C4', 'F3', 'F4', 'F7', 'F8', 'Fp1', 'Fp2', 'Fz', 'P7', 'P8', 'T7', 'T8']
    tiers = ['tier1', 'tier2', 'tier3']
    dataset_keys = ['torres_torres', 'ibarra_zarate', 'raeisi', 'wang']
    dataset_display = {'torres_torres': 'A', 'ibarra_zarate': 'B', 'raeisi': 'C', 'wang': 'D'}

    # Fixed tier-major column order as (tier, dataset_key) tuples
    col_order = [(tier, dk) for tier in tiers for dk in dataset_keys]
    x_labels = [f"T{tier[-1]}·{dataset_display[dk]}" for tier, dk in col_order]

    # Create a pivot table with proper structure
    pivot_df = df.pivot_table(
        index='channel', 
        columns=['tier', 'dataset'], 
        values='normalized_importance',
        aggfunc='mean'
    )
    
    # Reindex to ensure all channels are present in the correct order
    pivot_df = pivot_df.reindex(channels)
    
    # Reindex columns to match the expected order
    pivot_df = pivot_df.reindex(columns=col_order)
    
    # Convert to numpy matrix for plotting
    matrix = pivot_df.values
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    
    # Set ticks and labels
    ax.set_yticks(range(len(channels)))
    ax.set_yticklabels(channels)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels)
    
    # Annotate the cells with values
    for i in range(len(channels)):
        for j in range(len(x_labels)):
            val = matrix[i, j]
            text_color = "white" if val < 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=text_color)
    
    # Add colorbar
    fig.colorbar(im, ax=ax)
    
    # Set labels and title
    ax.set_xlabel("tier · dataset")
    ax.set_ylabel("channel")
    ax.set_title("Cross-tier channel importance (rank-normalized)")
    
    fig.tight_layout()
    
    # Save figure
    output_path = FIGURES_DIR / "tier23" / "interpretability_heatmap.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    print(f"Saved heatmap to {output_path}")

if __name__ == "__main__":
    main()