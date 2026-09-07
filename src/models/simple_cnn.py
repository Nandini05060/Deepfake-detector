import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    """
    Experiment 1 Baseline: 3-Stage ConvNet Reference.
    Input 224x224x3
    Conv -> BatchNorm -> ReLU -> MaxPool (Block 1)
    Conv -> BatchNorm -> ReLU -> MaxPool (Block 2)
    Conv -> BatchNorm -> ReLU -> MaxPool (Block 3)
    Global Average Pooling -> Fully Connected -> Binary Logit
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: Conv -> BN -> ReLU -> MaxPool
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2: Conv -> BN -> ReLU -> MaxPool
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3: Conv -> BN -> ReLU -> MaxPool
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Global Average Pooling
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes) # Binary Logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        logit = self.classifier(feat)
        return logit.squeeze(-1)

