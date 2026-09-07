import random
import numpy as np
import torch
import os
import sys

def seed_everything(seed: int = 42) -> dict:
    """
    Sets random seeds across Python, NumPy, PyTorch, and CUDA for total reproducibility.
    Returns metadata about the environment and random seed.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    info = {
        "seed": seed,
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A"
    }
    return info
