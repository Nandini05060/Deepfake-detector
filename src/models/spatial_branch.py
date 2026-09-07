import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class SpatialBranch(nn.Module):
    """
    Branch A — Spatial Domain Feature Extractor.
    Extracts spatial RGB embeddings F_s using ImageNet-pretrained EfficientNet.
    """
    def __init__(self, embed_dim: int = 256, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_b0(weights=weights)
        
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        
        backbone_dim = backbone.classifier[1].in_features # 1280
        
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        pooled = self.avgpool(feat)
        flattened = torch.flatten(pooled, 1)
        F_s = self.projection(flattened)
        return F_s
