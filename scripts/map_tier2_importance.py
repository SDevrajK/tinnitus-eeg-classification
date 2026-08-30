"""
Map the top Tier 2 importance features to physical interpretation (channel/band/feature-type).
"""
import pandas as pd
from _common import load_config, RESULTS_DIR

def parse_feature(name: str) -> tuple[str, str, str]:
    """
    Parse feature name into (feature_type, band, channel) tuple.
    
    Reproduces the Tier 1 coefficient-parsing convention from extract_coefficients.py
    (lines 17-36).
    
    Examples:
        power_alpha_C3 → ("power","alpha","C3")
        wpli_beta_F3_F4 → ("wpli","beta","F3-F4") 
        plzc_Cz → ("plzc","","Cz")
    """
    if name.startswith("power_"):
        parts = name.split("_")
        return ("power", parts[1], parts[2])
    elif name.startswith("wpli_"):
        parts = name.split("_")
        return ("wpli", parts[1], f"{parts[2]}-{parts[3]}")
    elif name.startswith("plzc_"):
        parts = name.split("_")
        return ("plzc", "", parts[1])
    else:
        raise ValueError(f"Unknown feature type for {name}")

def main():
    config = load_config()
    top_k = int(config["tier2"]["importance"]["top_k"])

    # Use real input path
    input_path = RESULTS_DIR / "tier2_feature_importance.csv"
    df = pd.read_csv(input_path)
    
    # Define the fixed dataset and model orders
    dataset_order = ["torres_torres", "ibarra_zarate", "raeisi", "wang"]
    model_order = ["random_forest", "svm"]
    
    rows = []
    
    for dataset in dataset_order:
        for model in model_order:
            # Filter data for this (dataset, model) group
            group_df = df[(df["dataset"] == dataset) & (df["model"] == model)]
            
            # Sort by importance descending (stable sort to ensure reproducibility)
            group_df = group_df.sort_values("importance", ascending=False, kind="stable")
            
            # Keep top_k rows
            top_df = group_df.head(top_k)
            
            # Apply parse_feature to each feature
            for rank, (_, row) in enumerate(top_df.iterrows(), 1):
                feature_type, band, channel = parse_feature(row["feature"])
                rows.append({
                    "dataset": dataset,
                    "model": model,
                    "rank": rank,
                    "feature": row["feature"],
                    "feature_type": feature_type,
                    "band": band,
                    "channel": channel,
                    "importance": row["importance"]
                })
    
    # Create output DataFrame
    output_df = pd.DataFrame(rows)
    
    # Ensure correct column order
    output_df = output_df[["dataset", "model", "rank", "feature", "feature_type", "band", "channel", "importance"]]
    
    # Write to CSV (no index)
    output_path = RESULTS_DIR / "tier2_top_features.csv"
    output_df.to_csv(output_path, index=False)
    print(f"Saved {len(output_df)} top features to {output_path}")



if __name__ == "__main__":
    main()