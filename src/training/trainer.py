import os
import time
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.training.losses import BinaryCrossEntropyLoss
from src.training.scheduler import CosineAnnealingWithWarmup
from src.evaluation.metrics import compute_all_metrics, find_optimal_threshold
from src.utils.logger import setup_logger

logger = setup_logger("LX-DFD.Trainer")

class Trainer:
    """
    Unified PyTorch Training Engine supporting:
    - Automatic Mixed Precision (torch.amp)
    - Cosine Annealing LR Scheduling with Warmup
    - Validation ROC-AUC Early Stopping & Best Model Checkpointing
    - Full Experiment Logging
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        epochs: int = 30,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 3,
        patience: int = 8,
        checkpoint_dir: str = "checkpoints",
        model_name: str = "lxfd_model",
        use_amp: bool = True,
        resume: bool = False,
        resume_path: str = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.patience = patience
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.use_amp = use_amp and (device.type == 'cuda')

        self.criterion = BinaryCrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = CosineAnnealingWithWarmup(
            self.optimizer,
            warmup_epochs=warmup_epochs,
            max_epochs=epochs
        )
        
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

        self.start_epoch = 1
        self.best_val_auc = 0.0
        self.history = []

        if resume or resume_path:
            ckpt_file = Path(resume_path) if resume_path else (self.checkpoint_dir / f"{self.model_name}_best.pt")
            if ckpt_file.exists():
                logger.info(f"Loading checkpoint for resumption: {ckpt_file}")
                checkpoint = torch.load(ckpt_file, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                if 'optimizer_state_dict' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.best_val_auc = checkpoint.get('val_auc', 0.0)

                hist_file = self.checkpoint_dir / f"{self.model_name}_history.json"
                if hist_file.exists():
                    try:
                        with open(hist_file, "r") as hf:
                            hdata = json.load(hf)
                            self.history = hdata.get("history", [])
                            self.start_epoch = len(self.history) + 1
                    except Exception as ex:
                        logger.warning(f"Could not load history file: {ex}")
                        self.start_epoch = checkpoint.get('epoch', 0) + 1
                else:
                    self.start_epoch = checkpoint.get('epoch', 0) + 1

                for _ in range(self.start_epoch - 1):
                    self.scheduler.step()

                logger.info(f"Resuming {self.model_name} from epoch {self.start_epoch}/{self.epochs} (Best Val AUC: {self.best_val_auc:.4f})")
            else:
                logger.warning(f"Resume requested but checkpoint {ckpt_file} not found. Starting from scratch.")

    def train_epoch(self) -> tuple:
        self.model.train()
        total_loss = 0.0
        all_targets = []
        all_probs = []

        for batch in self.train_loader:
            images = batch['image'].to(self.device, non_blocking=True)
            labels = batch['label'].to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    out = self.model(images)
                    logits = out[0] if isinstance(out, tuple) else out
                    loss = self.criterion(logits, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                out = self.model(images)
                logits = out[0] if isinstance(out, tuple) else out
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            total_loss += loss.item() * len(labels)
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(probs)

        avg_loss = total_loss / len(self.train_loader.dataset)
        metrics = compute_all_metrics(np.array(all_targets), np.array(all_probs))
        metrics['loss'] = float(avg_loss)
        return metrics

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> tuple:
        self.model.eval()
        total_loss = 0.0
        all_targets = []
        all_probs = []

        for batch in loader:
            images = batch['image'].to(self.device, non_blocking=True)
            labels = batch['label'].to(self.device, non_blocking=True)

            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    out = self.model(images)
                    logits = out[0] if isinstance(out, tuple) else out
                    loss = self.criterion(logits, labels)
            else:
                out = self.model(images)
                logits = out[0] if isinstance(out, tuple) else out
                loss = self.criterion(logits, labels)

            probs = torch.sigmoid(logits).cpu().numpy()
            total_loss += loss.item() * len(labels)
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(probs)

        avg_loss = total_loss / len(loader.dataset)
        y_true = np.array(all_targets)
        y_prob = np.array(all_probs)
        
        opt_thresh = find_optimal_threshold(y_true, y_prob)
        metrics = compute_all_metrics(y_true, y_prob, threshold=opt_thresh)
        metrics['loss'] = float(avg_loss)
        metrics['optimal_threshold'] = float(opt_thresh)
        return metrics, y_true, y_prob

    def train(self) -> dict:
        logger.info(f"Starting training for {self.model_name} on device {self.device} (AMP: {self.use_amp})...")
        best_val_auc = self.best_val_auc
        epochs_no_improve = 0
        history = list(self.history)
        start_time = time.time()

        for epoch in range(self.start_epoch, self.epochs + 1):
            epoch_start = time.time()
            
            train_metrics = self.train_epoch()
            val_metrics, _, _ = self.evaluate(self.val_loader)
            self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']
            epoch_time = time.time() - epoch_start

            logger.info(
                f"Epoch [{epoch:02d}/{self.epochs:02d}] "
                f"Train Loss: {train_metrics['loss']:.4f} | Train AUC: {train_metrics['roc_auc']:.4f} || "
                f"Val Loss: {val_metrics['loss']:.4f} | Val AUC: {val_metrics['roc_auc']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | LR: {current_lr:.6f} ({epoch_time:.1f}s)"
            )

            record = {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "lr": current_lr,
                "epoch_time": epoch_time
            }
            history.append(record)

            # Checkpoint best model based on validation ROC-AUC
            if val_metrics['roc_auc'] > best_val_auc:
                best_val_auc = val_metrics['roc_auc']
                epochs_no_improve = 0
                best_model_path = self.checkpoint_dir / f"{self.model_name}_best.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_auc': best_val_auc,
                    'val_metrics': val_metrics
                }, best_model_path)
                logger.info(f"  -> Saved new best model checkpoint to {best_model_path} (Val AUC: {best_val_auc:.4f})")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    logger.info(f"Early stopping triggered after {epoch} epochs (Patience {self.patience} reached).")
                    break

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f}s. Best Val ROC-AUC: {best_val_auc:.4f}")

        # Save training history JSON
        history_path = self.checkpoint_dir / f"{self.model_name}_history.json"
        with open(history_path, "w") as f:
            json.dump({
                "model_name": self.model_name,
                "best_val_auc": float(best_val_auc),
                "total_training_time": float(total_time),
                "history": history
            }, f, indent=2)

        return {
            "best_val_auc": float(best_val_auc),
            "history": history,
            "total_time": float(total_time),
            "checkpoint_path": str(self.checkpoint_dir / f"{self.model_name}_best.pt")
        }
