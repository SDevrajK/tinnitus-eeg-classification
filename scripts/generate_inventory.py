#!/usr/bin/env python3
"""
Metadata extractor for EEG files to generate inventory.
Reads only file headers without loading signal data into memory.
Uses scipy.io for .set files and numpy for .npz files as per PRD requirements.
"""

import scipy.io as sio
import numpy as np
import csv
from pathlib import Path

# Configurable constants
TARGET_ROOT = Path("/home/sdevrajk/media-hdd/researchdata/personal/MachineLearning/data/bids_organized")

# Sampling rate for Dataset C (raeisi) - documented value from Zenodo
RAEISI_SAMPLING_RATE = 1200.0

# Path to group manifest
MANIFEST_PATH = TARGET_ROOT / "subjects_groups.csv"

# Path to inventory output file
INVENTORY_PATH = TARGET_ROOT / "inventory.csv"

def extract_metadata(eeg_file_path) -> dict:
    """
    Return {"sfreq": float, "n_channels": int, "duration_sec": float} 
    read from the file header only.
    
    Uses scipy.io for .set files (EEGLAB format) and numpy for .npz files.
    Does NOT load signal data into memory.
    """
    suffix = eeg_file_path.suffix.lower()
    
    if suffix == ".npz":
        # Read only the shape via memory mapping - does NOT load array into RAM
        data = np.load(eeg_file_path, mmap_mode="r")
        arr = data["arr_0"]
        n_channels = arr.shape[0]  # 63
        n_samples = arr.shape[1]   # 360000
        sfreq = RAEISI_SAMPLING_RATE
        duration_sec = n_samples / sfreq
        return {"sfreq": sfreq, "n_channels": n_channels, "duration_sec": duration_sec}
    
    elif suffix == ".set":
        # Read MATLAB header with scipy.io - does NOT load signal data
        mat = sio.loadmat(eeg_file_path, struct_as_record=False, squeeze_me=True)
        
        # EEGLAB fields are stored in different layouts depending on dataset
        # Datasets A and D nest fields under a top-level "EEG" struct
        if "EEG" in mat:
            eeg = mat["EEG"]
            # Access attributes from the struct
            n_channels = int(eeg.nbchan)
            sfreq = float(eeg.srate)
            n_samples_total = int(eeg.trials) * int(eeg.pnts)
        # Dataset B stores fields FLATTENED at the top level
        else:
            # Direct access to dictionary keys
            n_channels = int(mat["nbchan"])
            sfreq = float(mat["srate"])
            n_samples_total = int(mat["trials"]) * int(mat["pnts"])
            
        duration_sec = n_samples_total / sfreq
        
        return {"sfreq": sfreq, "n_channels": n_channels, "duration_sec": duration_sec}
    
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def load_group_manifest() -> list[dict]:
    """
    Read MANIFEST_PATH with csv.DictReader. Return a list of dicts, one per subject,
    with keys exactly: "dataset", "subject_id", "group".
    There must be 441 entries.
    """
    subjects = []
    with open(MANIFEST_PATH, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            subjects.append({
                "dataset": row["Dataset"],
                "subject_id": row["Subject_ID"],
                "group": row["Group"]
            })
    return subjects


def locate_eeg_file(dataset: str, subject_id: str) -> Path:
    """
    Locate the EEG header file for a subject.
    
    eeg_dir = TARGET_ROOT / dataset / f"sub-{subject_id}" / "eeg"
    Return the single file in eeg_dir whose suffix (case-insensitive) is ".set" or ".npz".
    (Ignore ".fdt".) Raise ValueError if there is not exactly one such file.
    """
    eeg_dir = TARGET_ROOT / dataset / f"sub-{subject_id}" / "eeg"
    
    # Find all files with .set or .npz suffix (case insensitive)
    eeg_files = list(eeg_dir.glob("*.[sS][eE][tT]")) + list(eeg_dir.glob("*.[nN][pP][zZ]"))
    
    # Filter out .fdt files (they carry no header metadata)
    eeg_files = [f for f in eeg_files if f.suffix.lower() != ".fdt"]
    
    if len(eeg_files) != 1:
        raise ValueError(f"Expected exactly one EEG file for {dataset}/{subject_id}, got {len(eeg_files)} files: {eeg_files}")
    
    return eeg_files[0]


def build_inventory_rows(subjects: list[dict]) -> list[dict]:
    """
    For each subject dict from load_group_manifest(), locate its header file and 
    call extract_metadata on it. Build and return a list of row dicts with EXACTLY 
    these six keys (these are the inventory CSV column names):
    
    "Dataset"       -> subject["dataset"]
    "Subject_ID"    -> subject["subject_id"]
    "Group"         -> subject["group"]
    "Sampling_Rate" -> int(round(metadata["sfreq"]))     # e.g. 256, 512, 250, 1200
    "Channel_Count" -> metadata["n_channels"]            # e.g. 32, 16, 63, 118
    "Duration_sec"  -> round(metadata["duration_sec"], 1)  # one decimal place
    
    Return the list of 441 row dicts (preserve the manifest's order).
    """
    rows = []
    for subject in subjects:
        # Locate the EEG file for this subject
        eeg_file = locate_eeg_file(subject["dataset"], subject["subject_id"])
        
        # Extract metadata from the file
        metadata = extract_metadata(eeg_file)
        
        # Build the inventory row
        row = {
            "Dataset": subject["dataset"],
            "Subject_ID": subject["subject_id"],
            "Group": subject["group"],
            "Sampling_Rate": int(round(metadata["sfreq"])),
            "Channel_Count": metadata["n_channels"],
            "Duration_sec": round(metadata["duration_sec"], 1)
        }
        
        rows.append(row)
    
    return rows


def write_inventory(rows: list[dict], path) -> None:
    """
    Write a CSV file at `path` using csv.DictWriter.
    The fieldnames (column order and exact names) MUST be exactly:
        ["Dataset", "Subject_ID", "Group", "Sampling_Rate", "Channel_Count", "Duration_sec"]
    Open with newline="" to avoid blank lines; write the header row, then all rows.
    """
    fieldnames = ["Dataset", "Subject_ID", "Group", "Sampling_Rate", "Channel_Count", "Duration_sec"]
    
    with open(path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    subjects = load_group_manifest()
    rows = build_inventory_rows(subjects)
    write_inventory(rows, INVENTORY_PATH)
    print(f"Wrote {len(rows)} rows to {INVENTORY_PATH}")


if __name__ == "__main__":
    main()