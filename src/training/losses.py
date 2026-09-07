import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryCrossEntropyLoss(nn.Module):
    """
    Standard BCEWithLogitsLoss wrapper for stable binary classification.
    Calculates BCE directly on unnormalized logits.
    """
    def __init__(self, pos_weight: torch.Tensor = None):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.criterion(logits, targets)
