"""
Extract coefficients from trained elastic-net models and map them back to physical features.
"""

import numpy as np
import pandas as pd
import joblib
from _common import load_scaled_features, RESULTS_DIR

FEATURE_PREFIXES = ("power_", "wpli_", "plzc_")

def feature_columns(df):
    """Return list of column names that start with one of the feature prefixes."""
    return [col for col in df.columns if any(col.startswith(prefix) for prefix in FEATURE_PREFIXES)]

def parse_feature(name: str) -> tuple[str, str, str]:
    """
    Parse feature name into (feature_type, band, channel) tuple.
    
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
    MODELS_DIR = RESULTS_DIR / "models"
    
    rows = []
    
    # Dataset keys in order
    dataset_keys = ("torres_torres", "ibarra_zarate", "raeisi", "wang")
    
    for dataset_key in dataset_keys:
        # Load the scaled features to get the column names
        df = load_scaled_features(dataset_key)
        feat_cols = feature_columns(df)
        assert len(feat_cols) == 468
        
        # Load the trained model
        model = joblib.load(MODELS_DIR / f"{dataset_key}_elastic_net.joblib")
        
        # Extract coefficients
        coefs = np.asarray(model.coef_).ravel()
        assert len(coefs) == 468
        
        # Map coefficients back to features
        for i, fname in enumerate(feat_cols):
            ftype, band, channel = parse_feature(fname)
            rows.append({
                "dataset": dataset_key,
                "feature": fname,
                "feature_type": ftype,
                "band": band,
                "channel": channel,
                "coefficient": float(coefs[i])
            })
        
        # Print statistics
        K = int(np.sum(np.abs(coefs) > 1e-8))
        print(f"{dataset_key}: n_nonzero={K} (of 468)")
        
        # Print top 5 features by absolute coefficient
        top_indices = np.argsort(np.abs(coefs))[::-1][:5]
        top_features = []
        for idx in top_indices:
            fname = feat_cols[idx]
            coef = coefs[idx]
            top_features.append(f"{fname}={coef:+.4f}")
        print(f"top: {', '.join(top_features)}")
    
    # Write full table to CSV
    output_path = RESULTS_DIR / "elastic_net_coefficients.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Saved {len(rows)} coefficients to {output_path}")

if __name__ == "__main__":
    main()