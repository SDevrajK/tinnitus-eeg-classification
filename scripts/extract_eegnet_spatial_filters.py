"""
Extracts the EEGNet spatial-convolution filter weights from the trained Tier 3 models.
"""
import numpy as np
import torch
from _common import RESULTS_DIR

def main():
    MODELS_DIR = RESULTS_DIR / "models"
    spatial_filters = {}
    
    dataset_keys = ("torres_torres", "ibarra_zarate", "raeisi", "wang")
    
    for dataset_key in dataset_keys:
        # Load the state dictionary
        sd = torch.load(MODELS_DIR / f"{dataset_key}_eegnet.pt", weights_only=True)
        
        # Extract the spatial-convolution weight tensor
        weight_key = "conv_spatial.parametrizations.weight.original"
        weight_tensor = sd[weight_key]
        
        # Assert the raw shape is exactly (16, 1, 13, 1)
        assert weight_tensor.shape == (16, 1, 13, 1), f"Unexpected shape for {dataset_key}: {weight_tensor.shape}"
        
        # Reshape to (16, 13)
        w = weight_tensor.reshape(16, 13).detach().cpu().numpy().astype(np.float32)
        
        # Store in dict
        spatial_filters[dataset_key] = w
        
        # Print summary
        print(f"{dataset_key}: spatial filters shape {w.shape}")
    
    # Save the spatial filters
    torch.save(spatial_filters, RESULTS_DIR / "eegnet_spatial_filters.pt")
    print("Saved spatial filters to results/eegnet_spatial_filters.pt")

if __name__ == "__main__":
    main()