#!/usr/bin/env python3
"""
Exclude subjects whose retained recording duration is below the configurable minimum threshold.
"""
from pathlib import Path
import csv
import yaml

PROJECT_ROOT = Path("/home/sdevrajk/projects/personal/MachineLearning")
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
INVENTORY_PATH = Path("/home/sdevrajk/media-hdd/researchdata/personal/MachineLearning/data/bids_organized/inventory.csv")
DERIVATIVES_DIR = Path("/home/sdevrajk/media-hdd/researchdata/personal/MachineLearning/data/derivatives")
OUTPUT_PATH = DERIVATIVES_DIR / "excluded_subjects.csv"

def main():
    # Load config
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    
    threshold = float(config["min_duration_sec"])
    
    # Read inventory.csv and filter subjects
    excluded_subjects = []
    
    with open(INVENTORY_PATH, "r") as f:
        reader = csv.DictReader(f)
        total_subjects = 0
        
        for row in reader:
            total_subjects += 1
            subject_id = row["Subject_ID"]
            dataset = row["Dataset"]
            group = row["Group"]
            duration_sec = float(row["Duration_sec"])
            
            if duration_sec < threshold:
                excluded_subjects.append({
                    "Subject_ID": subject_id,
                    "Dataset": dataset,
                    "Group": group,
                    "Duration_sec": duration_sec,
                    "Threshold_sec": threshold
                })
    
    # Create derivatives directory if needed
    DERIVATIVES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Write excluded subjects to CSV
    with open(OUTPUT_PATH, "w", newline="") as f:
        fieldnames = ["Subject_ID", "Dataset", "Group", "Duration_sec", "Threshold_sec"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(excluded_subjects)
    
    # Print summary
    print(f"Total subjects read: {total_subjects}")
    print(f"Number excluded: {len(excluded_subjects)}")
    
    # Count by group
    excluded_by_group = {}
    for subject in excluded_subjects:
        group = subject["Group"]
        excluded_by_group[group] = excluded_by_group.get(group, 0) + 1
    
    print("Excluded subjects by group:")
    for group, count in excluded_by_group.items():
        print(f"  {group}: {count}")

if __name__ == "__main__":
    main()