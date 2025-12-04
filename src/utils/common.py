import torch
import random
import numpy as np

def get_device():
    """Returns the correct torch device"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed: int = 42):
    """Set all relevent random seeds for reproductibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False