import numpy as np
import torch

def get_num_samples(path):
    """Get number of samples in dataset."""
    arr = np.load(path, allow_pickle=True)
    return len(arr[0])

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
