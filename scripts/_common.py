"""Shared path constants and config/shared-channel loaders used by Phase 2+ scripts. Not a package — imported directly by scripts run from `scripts/`."""

from pathlib import Path
import yaml
import json

PROJECT_ROOT = Path("/home/sdevrajk/projects/personal/MachineLearning")
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
BIDS_ROOT = Path("/home/sdevrajk/media-hdd/researchdata/personal/MachineLearning/data/bids_organized")
DERIVATIVES_DIR = Path("/home/sdevrajk/media-hdd/researchdata/personal/MachineLearning/data/derivatives")
PREPROCESSED_DIR = DERIVATIVES_DIR / "preprocessed"
SHARED_CHANNELS_PATH = DERIVATIVES_DIR / "shared_channels.json"
INVENTORY_CSV = BIDS_ROOT / "inventory.csv"
FEATURES_DIR = Path("/home/sdevrajk/media-hdd/researchdata/personal/MachineLearning/data/features")


def load_config() -> dict:
    """Load the central configuration from YAML."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_shared_channels() -> list[str]:
    """Load the list of shared channels from JSON."""
    with open(SHARED_CHANNELS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["shared_channels"]
