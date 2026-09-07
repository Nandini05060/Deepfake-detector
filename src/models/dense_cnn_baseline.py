import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseLayer(nn.Module):
    """Dense block composite layer (BN-ReLU-Conv1x1 -> BN-ReLU-Conv3x3)."""
    def __init__(self, in_channels: int, growth_rate: int, bottleneck_factor: int = 4):
        super().__init__()
        inter_channels = bottleneck_factor * growth_rate
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layer(x)
        return torch.cat([x, out], 1)

class DenseBlock(nn.Module):
    """Dense Block containing multiple DenseLayers."""
    def __init__(self, num_layers: int, in_channels: int, growth_rate: int):
        super().__init__()
        layers = []
        channels = in_channels
        for i in range(num_layers):
            layers.append(DenseLayer(channels, growth_rate))
            channels += growth_rate
        self.block = nn.Sequential(*layers)
        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class TransitionLayer(nn.Module):
    """Transition layer between dense blocks (BN-ReLU-Conv1x1 -> AvgPool2x2)."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transition(x)

class DenseCNNBaseline(nn.Module):
    """
    Baseline 2: Dense CNN Architecture inspired by Patel et al.,
    "An Improved Dense CNN Architecture for Deepfake Image Detection", IEEE Access, 2023.
    
    IMPLEMENTATION BREAKDOWN:
    -------------------------
    * REPRODUCED:
      - Dense connectivity feature reuse pattern x_l = H_l([x_0, x_1, ..., x_{l-1}])
      - Composite bottleneck block structure: BN -> ReLU -> Conv1x1 -> BN -> ReLU -> Conv3x3
      - Transition pooling layer for feature map compression: BN -> ReLU -> Conv1x1 -> AvgPool2x2
    
    * ADAPTED:
      - Standardized growth rate k=32 and block depth configuration (6, 12, 24, 16)
      - Standard ImageNet input channel normalization and 224x224 input sizing
    
    * NEWLY IMPLEMENTED:
      - Binary classification single-logit output head with 0.4 dropout
      - Native PyTorch automatic mixed precision (AMP) compatibility
    """
    def __init__(
        self,
        growth_rate: int = 32,
        block_config: tuple = (6, 12, 24, 16),
        num_init_features: int = 64,
        num_classes: int = 1
    ):
        super().__init__()
        
        # Initial stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Dense blocks & transitions
        num_features = num_init_features
        self.features = nn.ModuleList()
        
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_layers=num_layers, in_channels=num_features, growth_rate=growth_rate)
            self.features.append(block)
            num_features = block.out_channels
            
            if i != len(block_config) - 1:
                trans_out = num_features // 2
                trans = TransitionLayer(in_channels=num_features, out_channels=trans_out)
                self.features.append(trans)
                num_features = trans_out
                
        self.final_bn = nn.BatchNorm2d(num_features)
        
        # Linear classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(num_features, num_classes) # Binary Logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)
        for layer in self.features:
            out = layer(out)
        out = F.relu(self.final_bn(out), inplace=True)
        logit = self.classifier(out)
        return logit.squeeze(-1)
