"""
CI smoke-test suite covering preprocessing / feature extraction / model training against the committed fixture (PRD FR3, AC2).
"""

import sys
from pathlib import Path
import yaml
import numpy as np
import mne
import pytest
import csv
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from preprocess_raw import preprocess_raw
from extract_power import extract_band_power
from extract_wpli import extract_wpli
from extract_plzc import extract_plzc
from _common import RANDOM_SEED
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score


@pytest.fixture
def config():
    """Load the project configuration portably."""
    config_path = REPO_ROOT / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def fixture_epochs():
    """Return a dict mapping subject_id -> loaded epochs."""
    epochs_dict = {}
    fixture_dir = REPO_ROOT / "tests" / "fixtures"
    fif_files = sorted(fixture_dir.glob("*.fif"))
    
    for p in fif_files:
        subject_id = p.stem
        epochs = mne.read_epochs(p, preload=True, verbose=False)
        epochs_dict[subject_id] = epochs
    
    assert epochs_dict, "No fixture epochs found"
    return epochs_dict


@pytest.fixture
def subjects_groups():
    """Read subjects_groups.csv and return a dict mapping subject_id -> group."""
    groups_path = REPO_ROOT / "tests" / "fixtures" / "subjects_groups.csv"
    groups_dict = {}
    with open(groups_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            groups_dict[row["Subject_ID"]] = row["Group"]
    return groups_dict


def test_preprocessing_stage_runs(fixture_epochs, config):
    """Test that preprocessing runs without raising unhandled exceptions."""
    # Use one subject's epochs for the test
    epochs = fixture_epochs["P01GA"]
    
    # Reconstruct a continuous Raw from the epochs
    data = epochs.get_data()  # shape (n_epochs, n_channels, n_samples)
    data = data.transpose(1, 0, 2).reshape(data.shape[1], -1)  # concatenate epochs
    info = mne.create_info(epochs.ch_names, sfreq=epochs.info["sfreq"], ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    
    # Run the preprocessing
    filtered = preprocess_raw(raw, config)
    
    # Assertions
    assert isinstance(filtered, mne.io.RawArray)
    assert filtered.get_data().shape[0] == len(epochs.ch_names)  # 13 channels
    assert np.isfinite(filtered.get_data()).all()


def test_feature_extraction_stage_runs(fixture_epochs, config):
    """Test that feature extraction runs without raising unhandled exceptions."""
    # Use one subject's epochs for the test
    epochs = fixture_epochs["P01GA"]
    
    # Derive shape expectations from the epochs
    n_epochs = len(epochs)
    n_channels = len(epochs.ch_names)
    n_pairs = n_channels * (n_channels - 1) // 2
    
    # Run the REAL feature-extraction stage on the fixture epochs
    power, power_cols = extract_band_power(epochs, config)
    wpli, wpli_cols = extract_wpli(epochs, config)
    plzc, plzc_cols = extract_plzc(epochs, config)
    
    # Assert with plain `assert` (no extra lib):
    assert power.shape == (n_epochs, n_channels * 5)
    assert len(power_cols) == n_channels * 5
    assert wpli.shape == (n_epochs, n_pairs * 5)
    assert len(wpli_cols) == n_pairs * 5
    assert plzc.shape == (n_epochs, n_channels)
    assert len(plzc_cols) == n_channels
    assert np.isfinite(power).all()
    assert np.isfinite(wpli).all()
    assert np.isfinite(plzc).all()


def test_model_training_stage_runs(fixture_epochs, subjects_groups, config):
    """Test that model training runs without raising unhandled exceptions."""
    # Build feature matrix for all subjects
    frames = []
    for subject_id, epochs in fixture_epochs.items():
        power, power_cols = extract_band_power(epochs, config)
        wpli, wpli_cols = extract_wpli(epochs, config)
        plzc, plzc_cols = extract_plzc(epochs, config)
        n_epochs = power.shape[0]
        meta = pd.DataFrame({
            "Subject_ID": [subject_id] * n_epochs,
            "Group": [subjects_groups[subject_id]] * n_epochs,
            "Epoch_Index": np.arange(n_epochs),
            "Dataset_ID": ["torres_torres"] * n_epochs
        })
        feat = pd.DataFrame(np.hstack([power, wpli, plzc]), columns=power_cols + wpli_cols + plzc_cols)
        frames.append(pd.concat([meta, feat], axis=1))
    
    df = pd.concat(frames, ignore_index=True)
    
    # Select feature columns by prefix
    feat_cols = [c for c in df.columns if c.startswith(("power_", "wpli_", "plzc_"))]
    X = df[feat_cols].to_numpy()
    y = (df["Group"] == "Tinnitus").astype(int).to_numpy()
    
    # Assertions
    assert X.shape[1] > 0
    assert len(y) == len(df)
    assert set(np.unique(y)) == {0, 1}
    
    # Construct and train classifier
    rf_cfg = config["tier2"]["random_forest"]
    clf = RandomForestClassifier(
        n_estimators=int(rf_cfg["n_estimators_grid"][0]),
        class_weight=rf_cfg["class_weight"],
        random_state=RANDOM_SEED
    )
    clf.fit(X, y)
    y_pred = clf.predict(X)
    acc = balanced_accuracy_score(y, y_pred)
    
    # Assertions
    assert y_pred.shape == y.shape
    assert 0.0 <= acc <= 1.0