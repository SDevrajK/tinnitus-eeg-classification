"""
Generate summary table showing feature count breakdown per dataset.
"""
import pandas as pd
from _common import FEATURES_DIR, PROJECT_ROOT

DATASET_KEYS = ("torres_torres", "ibarra_zarate", "raeisi", "wang")


def main():
    rows = []
    for dataset_key in DATASET_KEYS:
        df = pd.read_parquet(FEATURES_DIR / f"{dataset_key}_features.parquet")
        n_power = sum(c.startswith("power_") for c in df.columns)
        n_wpli = sum(c.startswith("wpli_") for c in df.columns)
        n_plzc = sum(c.startswith("plzc_") for c in df.columns)
        n_meta = len(df.columns) - n_power - n_wpli - n_plzc
        n_epochs = len(df)
        n_feature_cols = n_power + n_wpli + n_plzc
        n_total_cols = len(df.columns)
        rows.append({
            "dataset": dataset_key,
            "n_epochs": n_epochs,
            "n_power": n_power,
            "n_wpli": n_wpli,
            "n_plzc": n_plzc,
            "n_meta": n_meta,
            "n_feature_cols": n_feature_cols,
            "n_total_cols": n_total_cols,
        })
    
    df = pd.DataFrame(rows)
    print("Feature count breakdown per dataset")
    print(df)
    
    output_path = PROJECT_ROOT / "specs" / "tinnitus-eeg-interpretability" / "phase3" / "figures" / "7.1_feature_count_summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()