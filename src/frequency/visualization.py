import cv2
import numpy as np
import torch
from PIL import Image
from src.frequency.dct import DiscreteCosineTransform2D, extract_high_low_frequency_masks

def generate_dct_visualizations(image_np_rgb: np.ndarray, target_size: int = 224) -> dict:
    """
    Generates visual representations of 2D DCT spectrum and high-frequency maps.
    Input: RGB numpy array (H, W, 3), range [0, 255] or [0.0, 1.0].
    Returns base64/uint8 images for dashboard display.
    """
    if image_np_rgb.max() <= 1.0:
        image_np_rgb = (image_np_rgb * 255.0).astype(np.uint8)
        
    resized_rgb = cv2.resize(image_np_rgb, (target_size, target_size))
    tensor_in = torch.tensor(resized_rgb, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    
    dct_module = DiscreteCosineTransform2D(size=target_size, log_scale=True)
    with torch.no_grad():
        dct_spectrum = dct_module(tensor_in).squeeze().cpu().numpy() # (224, 224)
        
    # Generate low/high frequency masks
    low_mask, high_mask = extract_high_low_frequency_masks(size=target_size, cutoff_ratio=0.35)
    high_mask_np = high_mask.cpu().numpy()
    
    high_freq_spectrum = dct_spectrum * high_mask_np
    
    # Normalize spectrums to 0-255 uint8 colormaps (Jet/Inferno) for clear visual inspection
    dct_colormap = cv2.applyColorMap((dct_spectrum * 255.0).astype(np.uint8), cv2.COLORMAP_INFERNO)
    high_freq_colormap = cv2.applyColorMap((high_freq_spectrum * 255.0).astype(np.uint8), cv2.COLORMAP_JET)
    
    return {
        "original_rgb": resized_rgb,
        "dct_spectrum_gray": (dct_spectrum * 255.0).astype(np.uint8),
        "dct_colormap": cv2.cvtColor(dct_colormap, cv2.COLOR_BGR2RGB),
        "high_freq_colormap": cv2.cvtColor(high_freq_colormap, cv2.COLOR_BGR2RGB)
    }
