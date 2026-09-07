import torch
import logging

logger = logging.getLogger("LX-DFD")

def get_device() -> torch.device:
    """
    Detects hardware accelerator (CUDA / CPU) and returns torch.device.
    Logs detailed hardware information.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info(f"Using GPU device: {gpu_name} ({vram_gb:.2f} GB VRAM)")
        logger.info(f"PyTorch Version: {torch.__version__} | CUDA Version: {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        logger.info(f"CUDA unavailable. Falling back to CPU. PyTorch Version: {torch.__version__}")
    
    return device
