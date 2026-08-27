#!/usr/bin/env python3
"""
EEG Data Organization Script for BIDS-Style Directory Structure

This script discovers subjects from four raw EEG datasets and creates
the BIDS-style target directory skeleton for every subject.

Datasets:
- Dataset A: Torres-Torres et al. 2023 (Mendeley DOI 10.17632/fj7sskjdt7.5)
- Dataset B: Ibarra-Zárate et al. (Mendeley DOI 10.17632/kj443jc4yc)
- Dataset C: Raeisi et al. (Zenodo DOI 10.5281/zenodo.13308645)
- Dataset D: Wang et al. 2023 (IEEE JBHI)

This script implements discovery and directory creation logic only (subtask 1.1).
File copying and BIDS renaming are handled in later subtasks.
"""

import re
from pathlib import Path
import zipfile
import io
import shutil
import csv

# Configurable constants
SOURCE_ROOT = Path("/home/sdevrajk/media-hdd/researchdata/personal/MachineLearning/data/raw")
TARGET_ROOT = Path("/home/sdevrajk/media-hdd/researchdata/personal/MachineLearning/data/bids_organized")

# Dataset keys
DATASET_KEYS = ("torres_torres", "ibarra_zarate", "raeisi", "wang")


def discover_subjects(dataset_key: str) -> list[dict]:
    """Discover subjects from a specific dataset and return their information."""
    subjects = []
    
    if dataset_key == "torres_torres":
        # Dataset A: Torres-Torres et al.
        dataset_path = SOURCE_ROOT / "dataset_A_torres_torres"
        
        # Find all raw files with the pattern P<number>GAraw.set/P<number>GCraw.set
        # Pattern: P<number>{GA or GC}raw.{set or fdt}
        raw_pattern = re.compile(r"^(P\d+)(GA|GC)raw\.(set|fdt)$")
        
        # Find all .set files and match with corresponding .fdt
        set_files = list(dataset_path.rglob("*.set"))
        
        seen_subjects = set()
        for set_file in set_files:
            # Get full filename (with extension) to match pattern
            base_name = set_file.name
            
            # Match the pattern
            if raw_pattern.match(base_name):
                # If matched, extract components using regex
                match = raw_pattern.match(base_name)
                prefix = match.group(1)  # P<number> part  
                suffix = match.group(2)  # GA or GC part
                
                # Get group from suffix
                group = "Tinnitus" if suffix == "GA" else "Control"
                
                # Create subject ID (keep prefix as-is, including zero-padding)
                subject_id = f"{prefix}{suffix}"
                
                # Check if we've already seen this subject (by subject ID)
                if subject_id in seen_subjects:
                    continue
                    
                seen_subjects.add(subject_id)
                
                # Check if corresponding .fdt file exists
                fdt_file = set_file.with_suffix(".fdt")
                if not fdt_file.exists():
                    print(f"Warning: missing .fdt file for subject {subject_id}")
                    continue
                
                sources = [set_file, fdt_file]
                subjects.append({
                    "dataset": dataset_key,
                    "target_name": dataset_key,
                    "subject_id": subject_id,
                    "group": group,
                    "sources": sources
                })
    
    elif dataset_key == "ibarra_zarate":
        # Dataset B: Ibarra-Zárate et al.
        dataset_path = SOURCE_ROOT / "dataset_B_ibarra_zarate" / "Acoustic Therapies for Tinnitus Treatment An EEG Database" / "TA_Database_set"
        
        # Group folders: G1-Placebo, G2-BBT, G3-TRT, G4-EAE, G5-ADT, G6-Control
        group_folders = list(dataset_path.iterdir())
        seen_subjects = set()
        
        for group_folder in group_folders:
            if not group_folder.is_dir():
                continue
                
            # Get group from folder name
            group_folder_name = group_folder.name
            if "Control" in group_folder_name:
                group = "Control"
            else:
                group = "Tinnitus"
            
            # Get subject folders within the group folder (e.g., P1/, P2/, ...)
            subject_folders = list(group_folder.iterdir())
            
            for subject_folder in subject_folders:
                if not subject_folder.is_dir():
                    continue
                    
                # Look for baseline files like P1G6_Baseline_S1.set
                baseline_pattern = re.compile(r"P(\d+)G(\d+)_Baseline_S1\.set")
                baseline_files = list(subject_folder.glob("*_Baseline_S1.set"))
                
                for baseline_file in baseline_files:
                    match = baseline_pattern.match(baseline_file.name)
                    if not match:
                        continue
                    
                    # Get subject ID from filename (P<number>G<number>)
                    subject_id = f"P{match.group(1)}G{match.group(2)}"
                    
                    # Check if we've already seen this subject
                    if subject_id in seen_subjects:
                        continue
                        
                    seen_subjects.add(subject_id)
                    
                    # Find corresponding .fdt file
                    fdt_file = baseline_file.with_suffix(".fdt")
                    if not fdt_file.exists():
                        print(f"Warning: missing .fdt file for subject {subject_id} in dataset {dataset_key}")
                        continue
                    
                    sources = [baseline_file, fdt_file]
                    subjects.append({
                        "dataset": dataset_key,
                        "target_name": dataset_key,
                        "subject_id": subject_id,
                        "group": group,
                        "sources": sources
                    })
    
    elif dataset_key == "raeisi":
        # Dataset C: Raeisi et al.
        dataset_path = SOURCE_ROOT / "dataset_C_raeisi"
        
        # Files named H1.npz..H16.npz and T1.npz..T20.npz
        npz_files = list(dataset_path.glob("*.npz"))
        pattern = re.compile(r"^(H|T)(\d+)\.npz$")
        
        seen_subjects = set()
        for npz_file in npz_files:
            match = pattern.match(npz_file.name)
            if not match:
                continue
                
            prefix = match.group(1)  # H or T
            number = match.group(2)  # number part
            
            subject_id = f"{prefix}{number}"
            
            # Check if we've already seen this subject
            if subject_id in seen_subjects:
                continue
                
            seen_subjects.add(subject_id)
            
            # Get group from prefix
            group = "Control" if prefix == "H" else "Tinnitus"
            
            sources = [npz_file]
            subjects.append({
                "dataset": dataset_key,
                "target_name": dataset_key,
                "subject_id": subject_id,
                "group": group,
                "sources": sources
            })
    
    elif dataset_key == "wang":
        # Dataset D: Wang et al.
        dataset_path = SOURCE_ROOT / "dataset_D_wang"
        
        # Find the zip files
        zip_files = list(dataset_path.glob("*.zip"))
        
        seen_subjects = set()
        for zip_file in zip_files:
            # Check if this is tinnitus or control patient zip
            if "tinnitus" in zip_file.name.lower():
                group = "Tinnitus"
                prefix = "T"
            elif "eeg dataset" in zip_file.name.lower():
                group = "Control"
                prefix = "H"
            else:
                continue  # Skip unrecognized zip files
            
            # Extract zip contents to list the files
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_entries = zip_ref.namelist()
                
                if "tinnitus" in zip_file.name.lower():
                    # Process tinnitus zip entries - collect both .set and .fdt files
                    set_entries = {}
                    for entry in zip_entries:
                        if entry.endswith('.set') or entry.endswith('.fdt'):
                            # Extract subject number from entry name
                            set_name = Path(entry).name
                            match = re.match(r"(\d+)\.(set|fdt)$", set_name)
                            if match:
                                subject_num = match.group(1)
                                if subject_num not in set_entries:
                                    set_entries[subject_num] = {}
                                set_entries[subject_num][match.group(2)] = entry
                    
                    # Create subjects from pairs of .set and .fdt files
                    for subject_num, entries in set_entries.items():
                        if 'set' in entries and 'fdt' in entries:
                            subject_id = f"{prefix}{subject_num.zfill(3)}"  # Pad to 3 digits
                            
                            if subject_id in seen_subjects:
                                continue
                                
                            seen_subjects.add(subject_id)
                            
                            # Store both .set and .fdt entries
                            sources = [(zip_file, entries['set']), (zip_file, entries['fdt'])]
                            subjects.append({
                                "dataset": dataset_key,
                                "target_name": dataset_key,
                                "subject_id": subject_id,
                                "group": group,
                                "sources": sources
                            })
                else:
                    # Process control "eeg dataset" zip - first find the nested zip
                    nested_zip_name = None
                    for entry in zip_entries:
                        if entry.endswith(".zip"):
                            nested_zip_name = entry
                            break
                    
                    if nested_zip_name:
                        # Read the nested zip from bytes
                        nested_data = zip_ref.read(nested_zip_name)
                        with zipfile.ZipFile(io.BytesIO(nested_data), 'r') as nested_zip:
                            nested_entries = nested_zip.namelist()
                            
                            # Collect .set and .fdt entries by subject number
                            set_entries = {}
                            for entry in nested_entries:
                                if entry.endswith('.set') or entry.endswith('.fdt'):
                                    # Extract subject number from entry name
                                    set_name = Path(entry).name
                                    match = re.match(r"(\d+)\.(set|fdt)$", set_name)
                                    if match:
                                        subject_num = match.group(1)
                                        if subject_num not in set_entries:
                                            set_entries[subject_num] = {}
                                        set_entries[subject_num][match.group(2)] = entry
                            
                            # Create subjects from pairs of .set and .fdt files
                            for subject_num, entries in set_entries.items():
                                if 'set' in entries and 'fdt' in entries:
                                    subject_id = f"{prefix}{subject_num.zfill(3)}"  # Pad to 3 digits
                                    
                                    if subject_id in seen_subjects:
                                        continue
                                        
                                    seen_subjects.add(subject_id)
                                    
                                    # Store both .set and .fdt entries as 3-tuples 
                                    # (outer_zip_path, nested_entry_name, inner_entry_name)
                                    sources = [(zip_file, nested_zip_name, entries['set']), (zip_file, nested_zip_name, entries['fdt'])]
                                    subjects.append({
                                        "dataset": dataset_key,
                                        "target_name": dataset_key,
                                        "subject_id": subject_id,
                                        "group": group,
                                        "sources": sources
                                    })
    
    return subjects


def create_target_dirs(subjects: list[dict]) -> None:
    """Create target directory structure for each subject."""
    for subject in subjects:
        # Build target directory from subject data
        target_dir = TARGET_ROOT / subject["target_name"] / f"sub-{subject['subject_id']}" / "eeg"
        target_dir.mkdir(parents=True, exist_ok=True)


def copy_files(subjects: list[dict]) -> None:
    """Copy/extract source files to the target directories, preserving original filenames."""
    for subject in subjects:
        target_dir = TARGET_ROOT / subject["target_name"] / f"sub-{subject['subject_id']}" / "eeg"
        
        for source in subject["sources"]:
            # Derive original basename and BIDS name for idempotency checking
            if isinstance(source, Path):
                # Direct file path (datasets A/B/C)
                original_basename = source.name
            elif isinstance(source, tuple) and len(source) == 2:
                # 2-tuple for zip files (dataset D tinnitus subjects)
                original_basename = Path(source[1]).name
            elif isinstance(source, tuple) and len(source) == 3:
                # 3-tuple for nested zip files (dataset D healthy subjects)
                original_basename = Path(source[2]).name
            
            extension = Path(original_basename).suffix
            bids_name = f"sub-{subject['subject_id']}_task-rest_eeg{extension}"
            
            # Check if BIDS name already exists (idempotency check)
            if (target_dir / bids_name).exists():
                # Subject is already organized - skip this source
                continue
                
            if isinstance(source, Path):
                # Direct file path (datasets A/B/C)
                destination = target_dir / original_basename
                if not destination.exists():
                    shutil.copy2(source, destination)
            elif isinstance(source, tuple) and len(source) == 2:
                # 2-tuple for zip files (dataset D tinnitus subjects)
                zip_path, entry_name = source
                destination = target_dir / original_basename
                if not destination.exists():
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        file_bytes = zip_ref.read(entry_name)
                    with open(destination, 'wb') as f:
                        f.write(file_bytes)
            elif isinstance(source, tuple) and len(source) == 3:
                # 3-tuple for nested zip files (dataset D healthy subjects)
                outer_zip_path, nested_entry_name, inner_entry_name = source
                destination = target_dir / original_basename
                if not destination.exists():
                    with zipfile.ZipFile(outer_zip_path, 'r') as outer_zip:
                        nested_data = outer_zip.read(nested_entry_name)
                    with zipfile.ZipFile(io.BytesIO(nested_data), 'r') as inner_zip:
                        file_bytes = inner_zip.read(inner_entry_name)
                    with open(destination, 'wb') as f:
                        f.write(file_bytes)


def rename_to_bids(subjects: list[dict]) -> None:
    """Rename subject files to BIDS convention: sub-<ID>_task-rest_eeg.<ext>."""
    for subject in subjects:
        target_dir = TARGET_ROOT / subject["target_name"] / f"sub-{subject['subject_id']}" / "eeg"
        
        for source in subject["sources"]:
            # Derive original basename, extension, and BIDS name for idempotency
            if isinstance(source, Path):
                # Direct file path (datasets A/B/C)
                original_basename = source.name
            elif isinstance(source, tuple) and len(source) == 2:
                # 2-tuple for zip files (dataset D tinnitus subjects)
                original_basename = Path(source[1]).name
            elif isinstance(source, tuple) and len(source) == 3:
                # 3-tuple for nested zip files (dataset D healthy subjects)
                original_basename = Path(source[2]).name
            
            extension = Path(original_basename).suffix
            bids_name = f"sub-{subject['subject_id']}_task-rest_eeg{extension}"
            
            # Clean up: if BIDS name already exists, remove any original duplicates
            original_path = target_dir / original_basename
            new_path = target_dir / bids_name
            
            if new_path.exists():
                # Already renamed - remove any leftover original duplicate
                if original_path.exists() and original_path != new_path:
                    original_path.unlink()
            elif original_path.exists():
                # Rename original to BIDS name
                original_path.rename(new_path)


def write_group_manifest(subjects: list[dict]) -> None:
    """Write a CSV manifest file with subject groups for inventory purposes."""
    manifest_path = TARGET_ROOT / "subjects_groups.csv"
    
    # Open with newline="" to avoid blank lines
    with open(manifest_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(["Dataset", "Subject_ID", "Group"])
        
        # Write subject rows
        for subject in subjects:
            writer.writerow([subject["target_name"], subject["subject_id"], subject["group"]])


def main() -> None:
    """Main function that discovers subjects and creates target directories for all datasets."""
    total_subjects = 0
    total_tinnitus = 0
    total_control = 0
    
    # Collect all subjects across datasets to build complete manifest
    all_subjects = []
    
    for key in DATASET_KEYS:
        subjects = discover_subjects(key)
        create_target_dirs(subjects)
        copy_files(subjects)
        rename_to_bids(subjects)
        all_subjects.extend(subjects)
        
        # Count by group
        tinnitus_count = sum(1 for s in subjects if s["group"] == "Tinnitus")
        control_count = sum(1 for s in subjects if s["group"] == "Control")
        
        print(f"Dataset {key}: {len(subjects)} subjects ({tinnitus_count} Tinnitus, {control_count} Control)")
        
        total_subjects += len(subjects)
        total_tinnitus += tinnitus_count
        total_control += control_count
    
    # Write the complete manifest file once with all subjects
    write_group_manifest(all_subjects)
    
    print(f"Total: {total_subjects} subjects ({total_tinnitus} Tinnitus, {total_control} Control)")


if __name__ == "__main__":
    main()