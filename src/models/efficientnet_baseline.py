import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class EfficientNetBaseline(nn.Module):
    """
    Baseline 3: Modern Transfer Learning CNN Baseline (EfficientNet-B0 pretrained on ImageNet).
    """
    def __init__(self, pretrained: bool = True, num_classes: int = 1, dropout: float = 0.3):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_b0(weights=weights)
        
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        
        in_features = backbone.classifier[1].in_features # 1280 for EfficientNet-B0
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        pooled = self.avgpool(feat)
        flattened = torch.flatten(pooled, 1)
        logit = self.classifier(flattened)
        return logit.squeeze(-1)
