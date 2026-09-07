import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def generate_dct_matrix(N: int) -> torch.Tensor:
    """
    Generates orthogonal NxN DCT-II transformation matrix C.
    C_ij = sqrt(1/N) for i=0, else sqrt(2/N) * cos(pi * (2j + 1) * i / (2N)).
    DCT2D(X) = C @ X @ C.T
    """
    C = torch.zeros((N, N), dtype=torch.float32)
    for i in range(N):
        for j in range(N):
            if i == 0:
                C[i, j] = 1.0 / math.sqrt(N)
            else:
                C[i, j] = math.sqrt(2.0 / N) * math.cos((math.pi * (2 * j + 1) * i) / (2.0 * N))
    return C

class DiscreteCosineTransform2D(nn.Module):
    """
    PyTorch 2D DCT-II Layer (Deterministic & GPU-accelerated).
    Converts RGB / Grayscale Image Tensors (B, C, H, W) to 2D DCT Log-Magnitude representations.
    """
    def __init__(self, size: int = 224, log_scale: bool = True, eps: float = 1e-6):
        super().__init__()
        self.size = size
        self.log_scale = log_scale
        self.eps = eps
        
        # Register DCT basis matrix C as buffer (non-trainable)
        C = generate_dct_matrix(size)
        self.register_buffer('C', C) # (N, N)
        self.register_buffer('C_T', C.t()) # (N, N)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: Tensor of shape (B, C, H, W) or (B, H, W)
        Output: 2D DCT Log-Magnitude Spectrum of shape (B, C, H, W)
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        B, C, H, W = x.shape
        
        if H != self.size or W != self.size:
            x = F.interpolate(x, size=(self.size, self.size), mode='bilinear', align_corners=False)
            
        # Convert RGB to Luminance (Y) channel if C == 3:
        # Y = 0.299 R + 0.587 G + 0.114 B
        if C == 3:
            r, g, b = x[:, 0:1, :, :], x[:, 1:2, :, :], x[:, 2:3, :, :]
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
        else:
            luminance = x
            
        # Compute 2D DCT via Matrix Multiplication: Y = C @ X @ C.T
        # Batch matrix multiplication: (B, 1, N, N)
        dct_map = torch.matmul(self.C, torch.matmul(luminance, self.C_T))
        
        # Log-magnitude normalization: log(1 + |DCT|)
        if self.log_scale:
            dct_map = torch.log(1.0 + torch.abs(dct_map) + self.eps)
            # Min-Max normalization per sample to [0, 1]
            min_val = dct_map.view(B, -1).min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
            max_val = dct_map.view(B, -1).max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
            dct_map = (dct_map - min_val) / (max_val - min_val + self.eps)
            
        return dct_map

def extract_high_low_frequency_masks(size: int = 224, cutoff_ratio: float = 0.3):
    """
    Creates circular or rectangular low-pass and high-pass masks for 2D DCT spectrum analysis.
    In DCT, top-left is low frequency, bottom-right is high frequency.
    """
    low_mask = torch.zeros((size, size), dtype=torch.float32)
    high_mask = torch.ones((size, size), dtype=torch.float32)
    
    radius = int(size * cutoff_ratio)
    for i in range(size):
        for j in range(size):
            if i + j <= radius: # Low-frequency region (top-left triangle)
                low_mask[i, j] = 1.0
                high_mask[i, j] = 0.0
                
    return low_mask, high_mask
