import sys
import json
import argparse
from pathlib import Path
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.dataset import DeepfakeDataset
from src.models.simple_cnn import SimpleCNN
from src.models.dense_cnn_baseline import DenseCNNBaseline
from src.models.efficientnet_baseline import EfficientNetBaseline
from src.models.lxfd_model import LXDFDModel
from src.evaluation.metrics import compute_all_metrics
from src.utils.device import get_device

def main():
    parser = argparse.ArgumentParser(description="Evaluate Trained Deepfake Detection Models")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint .pt file")
    parser.add_argument("--model-type", type=str, default="lxfd", choices=["simple_cnn", "dense_cnn", "efficientnet", "lxfd"])
    parser.add_argument("--fusion", type=str, default="attention")
    parser.add_argument("--split-csv", type=str, default="data/splits/test.csv")
    args = parser.parse_args()

    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    device = get_device()
    test_csv = Path(args.split_csv)
    ckpt_path = Path(args.checkpoint)

    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    if not test_csv.exists():
        print(f"[ERROR] Test CSV split not found: {test_csv}")
        sys.exit(1)

    test_dataset = DeepfakeDataset(str(test_csv), split='test')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Instantiate model
    if args.model_type == "simple_cnn":
        model = SimpleCNN()
    elif args.model_type == "dense_cnn":
        model = DenseCNNBaseline()
    elif args.model_type == "efficientnet":
        model = EfficientNetBaseline(pretrained=False)
    elif args.model_type == "lxfd":
        model = LXDFDModel(fusion_type=args.fusion, pretrained=False)

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    all_targets, all_probs = [], []
    with torch.no_grad():
        for batch in test_loader:
            imgs = batch['image'].to(device)
            lbls = batch['label'].to(device)
            out = model(imgs)
            logits = out[0] if isinstance(out, tuple) else out
            probs = torch.sigmoid(logits).cpu().numpy()
            all_targets.extend(lbls.cpu().numpy())
            all_probs.extend(probs)

    metrics = compute_all_metrics(np.array(all_targets), np.array(all_probs))
    
    out_dir = Path("reports/evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{ckpt_path.stem}_eval_metrics.json"
    
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[EVALUATION] Evaluated {ckpt_path.name}:")
    print(f"  - Accuracy: {metrics['accuracy']:.4f}")
    print(f"  - Precision: {metrics['precision']:.4f}")
    print(f"  - Recall: {metrics['recall']:.4f}")
    print(f"  - F1 Score: {metrics['f1']:.4f}")
    print(f"  - ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"  - Brier Score: {metrics['brier_score']:.4f}")
    print(f"  - Metrics saved to {out_file}")

if __name__ == "__main__":
    main()
