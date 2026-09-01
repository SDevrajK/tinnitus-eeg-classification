"""
Test fixture generation script for Dataset A.
Creates truncated epoch files from preprocessed data for reproducible testing.
See PRD §5 FR1-2, AC1, AC7.
"""
import csv
from pathlib import Path

import mne

from _common import PREPROCESSED_DIR, INVENTORY_CSV, PROJECT_ROOT

# Fixed inputs per specification
FIXTURE_SUBJECTS = {"torres_torres": ["P01GA", "P13GC"]}
N_EPOCHS = 5
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"


def main():
    # Create fixtures directory
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Build subject -> (dataset, group) lookup from inventory.csv
    subject_groups = {}
    with open(INVENTORY_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            subject_groups[row["Subject_ID"]] = (row["Dataset"], row["Group"])
    
    # Process each subject
    for dataset_key, subject_ids in FIXTURE_SUBJECTS.items():
        for subject_id in subject_ids:
            # Source path
            src_path = PREPROCESSED_DIR / dataset_key / f"{subject_id}.fif"
            
            # Check that source file exists
            if not src_path.exists():
                raise FileNotFoundError(f"Source file not found: {src_path}")
            
            # Load epochs
            epochs = mne.read_epochs(src_path, preload=True, verbose=False)
            
            # Truncate to first N epochs
            epochs = epochs[:N_EPOCHS]
            
            # Output path
            out_path = FIXTURES_DIR / f"{subject_id}.fif"
            
            # Save truncated epochs
            epochs.save(out_path, overwrite=True)
            
            # Print progress
            print(f"{dataset_key} {subject_id}: saved {len(epochs)} epochs -> {out_path}")
    
    # Write manifest file
    with open(FIXTURES_DIR / "subjects_groups.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Dataset", "Subject_ID", "Group"])
        writer.writeheader()
        rows_written = 0
        for dataset_key, subject_ids in FIXTURE_SUBJECTS.items():
            for subject_id in subject_ids:
                dataset, group = subject_groups[subject_id]
                writer.writerow({"Dataset": dataset, "Subject_ID": subject_id, "Group": group})
                rows_written += 1
        print(f"Manifest written: {rows_written} rows")


if __name__ == "__main__":
    main()