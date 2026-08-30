"""
Computes Integrated-Gradients saliency maps for the trained EEGNet models,
aggregated over a sample of epochs.
"""
import numpy as np
import torch
from captum.attr import IntegratedGradients
from _common import load_config, RANDOM_SEED, RESULTS_DIR
from eegnet_data import build_dataset
from train_eegnet import build_model

def main():
    config = load_config()
    eg = config["tier3"]["eegnet"]
    interp = config["tier3"]["interpretability"]
    ig_n_steps = int(interp["ig_n_steps"])
    n_saliency_epochs = int(interp["n_saliency_epochs"])
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Seed for the epoch sampling
    rng = np.random.default_rng(RANDOM_SEED)
    
    MODELS_DIR = RESULTS_DIR / "models"
    
    # Dataset keys, fixed order
    dataset_keys = ("torres_torres", "ibarra_zarate", "raeisi", "wang")
    
    saliency_maps = {}
    
    for dataset_key in dataset_keys:
        # Load the dataset
        X, y, subject_ids = build_dataset(dataset_key)
        n_chans, n_times = X.shape[1], X.shape[2]
        
        # Rebuild the model
        model = build_model(n_chans, n_times, eg)
        model.load_state_dict(torch.load(MODELS_DIR / f"{dataset_key}_eegnet.pt", weights_only=True))
        model.to(device)
        model.eval()
        
        # Sample epochs for saliency
        # Choose n_saliency_epochs indices without replacement
        idx = rng.choice(len(X), size=n_saliency_epochs, replace=False)
        idx = np.sort(idx)
        
        # Build the input batch
        inputs = torch.tensor(X[idx], dtype=torch.float32).to(device)  # shape (n_saliency_epochs, n_chans, n_times)
        
        # Compute Integrated Gradients
        # target=1 is the tinnitus class logit
        ig = IntegratedGradients(model)
        attributions = ig.attribute(inputs, target=1, n_steps=ig_n_steps)
        # attributions has shape (n_saliency_epochs, n_chans, n_times)
        
        # Aggregate: average over the sampled epochs
        saliency = np.abs(attributions.detach().cpu().numpy()).mean(axis=0)  # shape (n_chans, n_times)
        saliency = saliency.astype(np.float32)
        
        saliency_maps[dataset_key] = saliency
        
        print(f"{dataset_key}: saliency shape {saliency.shape}")
    
    # Save saliency maps
    torch.save(saliency_maps, RESULTS_DIR / "eegnet_saliency.pt")

if __name__ == "__main__":
    main()