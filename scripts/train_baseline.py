import sys
import argparse
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader

# Ensure root is on path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.dataset import DeepfakeDataset
from src.models.simple_cnn import SimpleCNN
from src.models.dense_cnn_baseline import DenseCNNBaseline
from src.models.efficientnet_baseline import EfficientNetBaseline
from src.training.trainer import Trainer
from src.utils.seed import seed_everything
from src.utils.device import get_device

def main():
    parser = argparse.ArgumentParser(description="Train Baseline Deepfake Models")
    parser.add_argument("--model", type=str, default="simple_cnn", choices=["simple_cnn", "dense_cnn", "efficientnet"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--resume", action="store_true", help="Resume training from existing checkpoint")
    parser.add_argument("--resume-path", type=str, default=None, help="Explicit checkpoint file to resume from")
    args = parser.parse_args()

    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    seed_everything(config['training']['seed'])
    device = get_device()

    splits_dir = Path(config['data']['splits_dir'])
    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"

    if not train_csv.exists() or not val_csv.exists():
        print("[ERROR] Split files not found. Please run scripts/create_splits.py first!")
        sys.exit(1)

    train_dataset = DeepfakeDataset(str(train_csv), split='train')
    val_dataset = DeepfakeDataset(str(val_csv), split='val')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Initialize model architecture
    if args.model == "simple_cnn":
        model = SimpleCNN()
        name = "simple_cnn"
    elif args.model == "dense_cnn":
        model = DenseCNNBaseline()
        name = "dense_cnn"
    elif args.model == "efficientnet":
        model = EfficientNetBaseline(pretrained=True)
        name = "efficientnet"

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        patience=config['training']['early_stopping_patience'],
        checkpoint_dir="checkpoints",
        model_name=name,
        use_amp=config['training']['use_amp'],
        resume=args.resume,
        resume_path=args.resume_path
    )

    results = trainer.train()
    print(f"[SUCCESS] {name} trained! Best Val ROC-AUC: {results['best_val_auc']:.4f}")

if __name__ == "__main__":
    main()
