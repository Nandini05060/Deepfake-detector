import math
import torch
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingWithWarmup(_LRScheduler):
    """
    Cosine Annealing LR Scheduler with Linear Warmup phase.
    """
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        min_lr: float = 1e-6,
        last_epoch: int = -1
    ):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear Warmup
            alpha = float(self.last_epoch + 1) / float(max(1, self.warmup_epochs))
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine Annealing
            progress = float(self.last_epoch - self.warmup_epochs) / float(max(1, self.max_epochs - self.warmup_epochs))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [self.min_lr + (base_lr - self.min_lr) * cosine_decay for base_lr in self.base_lrs]
