"""Apply robust scaling (median + IQR) to the engineered feature matrices."""

import numpy as np
from sklearn.preprocessing import RobustScaler
from _common import SCALED_FEATURES_DIR, load_config, load_features

FEATURE_PREFIXES = ("power_", "wpli_", "plzc_")

def feature_columns(df) -> list[str]:
    """Return all column names that start with one of the feature prefixes."""
    return [col for col in df.columns if any(col.startswith(prefix) for prefix in FEATURE_PREFIXES)]

def main():
    config = load_config()
    qrange = tuple(config["tier1"]["robust_scaler"]["quantile_range"])
    SCALED_FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    datasets = ("torres_torres", "ibarra_zarate", "raeisi", "wang")
    
    for dataset_key in datasets:
        df = load_features(dataset_key)
        feat_cols = feature_columns(df)
        assert len(feat_cols) == 468
        
        # Fit scaler independently for each dataset
        scaler = RobustScaler(quantile_range=qrange)
        scaler.fit(df[feat_cols])
        
        scaled = scaler.transform(df[feat_cols])
        
        # Build output DataFrame preserving meta columns
        out = df.copy()
        out[feat_cols] = scaled
        
        # Write scaled features to parquet
        out.to_parquet(SCALED_FEATURES_DIR / f"{dataset_key}_scaled.parquet", index=False)
        
        # Print sanity check
        meds = np.median(out[feat_cols], axis=0)
        iqrs = np.percentile(out[feat_cols], 75, axis=0) - np.percentile(out[feat_cols], 25, axis=0)
        print(f"{dataset_key}: n_epochs={len(df)}, median-of-medians={float(np.median(meds)):.4f}, median-IQR={float(np.median(iqrs)):.4f} -> {SCALED_FEATURES_DIR / f'{dataset_key}_scaled.parquet'}")

if __name__ == "__main__":
    main()