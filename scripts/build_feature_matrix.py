"""Phase 3 feature matrix assembly: concatenate band power, wPLI, PLZC per epoch, add metadata, save Parquet per dataset."""

import csv
import time
import numpy as np
import pandas as pd
from _common import FEATURES_DIR, INVENTORY_CSV, load_config
from load_epochs import load_dataset_epochs
from extract_power import extract_band_power
from extract_wpli import extract_wpli
from extract_plzc import extract_plzc

DATASET_KEYS = ("torres_torres", "ibarra_zarate", "raeisi", "wang")

def load_group_map() -> dict[str, str]:
    """Load subject_id -> Group mapping from inventory.csv."""
    with INVENTORY_CSV.open("r") as f:
        reader = csv.DictReader(f)
        return {row["Subject_ID"]: row["Group"] for row in reader}

def build_dataset_features(dataset_key: str, config: dict, group_map: dict) -> pd.DataFrame:
    """Build one row per epoch: metadata + band-power + wPLI + PLZC features for a dataset."""
    epochs_by_subject = load_dataset_epochs(dataset_key)
    frames = []
    for subject_id, epochs in sorted(epochs_by_subject.items()):
        power, power_cols = extract_band_power(epochs, config)
        wpli, wpli_cols = extract_wpli(epochs, config)
        plzc, plzc_cols = extract_plzc(epochs, config)
        n_epochs = power.shape[0]
        meta = pd.DataFrame({
            "Subject_ID": [subject_id] * n_epochs,
            "Group": [group_map[subject_id]] * n_epochs,
            "Epoch_Index": np.arange(n_epochs),
            "Dataset_ID": [dataset_key] * n_epochs
        })
        feat = pd.DataFrame(np.hstack([power, wpli, plzc]), columns=power_cols + wpli_cols + plzc_cols)
        frames.append(pd.concat([meta, feat], axis=1))
    return pd.concat(frames, ignore_index=True)

def main() -> None:
    config = load_config()
    group_map = load_group_map()
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    for dataset_key in DATASET_KEYS:
        t0 = time.time()
        df = build_dataset_features(dataset_key, config, group_map)
        elapsed = time.time() - t0
        out_path = FEATURES_DIR / f"{dataset_key}_features.parquet"
        df.to_parquet(out_path, index=False)
        n_feat = sum(1 for c in df.columns if c.startswith(("power_", "wpli_", "plzc_")))
        print(f"{dataset_key}: epochs={len(df)}, feature_columns={n_feat}, total_columns={len(df.columns)}, time={elapsed:.1f}s -> {out_path}")

if __name__ == "__main__":
    main()