#!/usr/bin/env python3
"""
Compute the intersection of standardized channel names across all four datasets
and write shared_channels.json with the final shared channel list and count.
"""

import json

from _common import CONFIG_PATH, DERIVATIVES_DIR, load_config

# Named constants
OUTPUT_PATH = DERIVATIVES_DIR / "shared_channels.json"
DATASET_KEYS = ("torres_torres", "ibarra_zarate", "raeisi", "wang")

def main():
    # Load config
    config = load_config()
    
    channel_mapping = config["channel_mapping"]
    
    # For each dataset key, derive the set of CANONICAL channel names
    dataset_channels = []
    
    for dataset_key in DATASET_KEYS:
        dataset_entry = channel_mapping[dataset_key]
        
        # Handle different data structures (dict vs list)
        if isinstance(dataset_entry, dict):
            # For torres_torres, ibarra_zarate, and wang
            canonical_names = set(dataset_entry.values())
        elif isinstance(dataset_entry, list):
            # For raeisi
            canonical_names = set(dataset_entry)
        else:
            raise ValueError(f"Unexpected data type for {dataset_key}: {type(dataset_entry)}")
        
        dataset_channels.append(canonical_names)
    
    # Compute the intersection of these four sets
    intersection = set.intersection(*dataset_channels)
    
    # Sort the intersection deterministically
    sorted_channels = sorted(intersection)
    
    # Write shared_channels.json
    DERIVATIVES_DIR.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "shared_channels": sorted_channels,
        "count": len(sorted_channels)
    }
    
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    
    # Print summary to stdout
    channels_str = " ".join(sorted_channels)
    print(f"Shared channels ({len(sorted_channels)}): {channels_str}")

if __name__ == "__main__":
    main()