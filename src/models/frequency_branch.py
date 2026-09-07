import torch
import torch.nn as nn
import torch.nn.functional as F
from src.frequency.dct import DiscreteCosineTransform2D

class FrequencyBranch(nn.Module):
    """
    Branch B — Frequency Domain Feature Extractor.
    Computes 2D DCT log-magnitude spectrum and extracts frequency artifact embeddings F_f.
    """
    def __init__(self, in_size: int = 224, embed_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.dct_transform = DiscreteCosineTransform2D(size=in_size, log_scale=True)
        
        # Frequency CNN (Specialized 4-Layer ConvNet for spectral patterns)
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2), # (112, 112)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # (56, 56)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # (28, 28)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # (14, 14)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.projection = nn.Sequential(
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Input: RGB Tensor (B, 3, H, W)
        Returns: Tuple of (F_f embedding, dct_map spectrum tensor)
        """
        dct_map = self.dct_transform(x) # (B, 1, H, W)
        feat = self.features(dct_map)
        flattened = torch.flatten(feat, 1)
        F_f = self.projection(flattened)
        return F_f, dct_map
