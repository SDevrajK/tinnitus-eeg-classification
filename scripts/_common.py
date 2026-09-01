"""Shared path constants and config/shared-channel loaders used by Phase 2+ scripts. Not a package — imported directly by scripts run from `scripts/`."""

from pathlib import Path
import yaml
import json
import pandas as pd

# Repo root derived from this file's location — portable across machines and inside the Docker container.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config() -> dict:
    """Load the central configuration from YAML."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Data root is the only machine-specific path; it lives in config.yaml (single source of truth).
# BIDS, derivatives, and features all derive from it.
_DATA_ROOT = Path(load_config()["paths"]["data_root"])
BIDS_ROOT = _DATA_ROOT / "bids_organized"
DERIVATIVES_DIR = _DATA_ROOT / "derivatives"
PREPROCESSED_DIR = DERIVATIVES_DIR / "preprocessed"
SHARED_CHANNELS_PATH = DERIVATIVES_DIR / "shared_channels.json"
INVENTORY_CSV = BIDS_ROOT / "inventory.csv"
FEATURES_DIR = _DATA_ROOT / "features"
SCALED_FEATURES_DIR = FEATURES_DIR / "scaled"


def load_shared_channels() -> list[str]:
    """Load the list of shared channels from JSON."""
    with open(SHARED_CHANNELS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["shared_channels"]

# Phase 4: fixed seed for permutation tests, bootstrap resampling, and model init (PRD §6)
RANDOM_SEED = 42

# Phase 4: seed for subject-stratified epoch capping in LODO training (PRD §6, configurable)
EPOCH_DROP_SEED = 42

PHASE4_DIR = PROJECT_ROOT / "specs" / "tinnitus-eeg-interpretability" / "phase4"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

def load_features(dataset_key: str) -> "pd.DataFrame":
    """Load full feature matrix for a dataset."""
    path = FEATURES_DIR / f"{dataset_key}_features.parquet"
    return pd.read_parquet(path)


def load_scaled_features(dataset_key: str) -> "pd.DataFrame":
    """Load the robust-scaled feature matrix for one dataset (full DataFrame incl. meta cols)."""
    return pd.read_parquet(SCALED_FEATURES_DIR / f"{dataset_key}_scaled.parquet")
