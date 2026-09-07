import sys
import json
import argparse
from pathlib import Path
import yaml
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.dataset import DeepfakeDataset
from src.models.simple_cnn import SimpleCNN
from src.models.dense_cnn_baseline import DenseCNNBaseline
from src.models.efficientnet_baseline import EfficientNetBaseline
from src.models.lxfd_model import LXDFDModel
from src.evaluation.robustness import evaluate_robustness
from src.utils.device import get_device

def main():
    parser = argparse.ArgumentParser(description="Run Transformation Robustness Benchmark")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--model-type", type=str, default="lxfd", choices=["simple_cnn", "dense_cnn", "efficientnet", "lxfd"])
    parser.add_argument("--fusion", type=str, default="attention")
    args = parser.parse_args()

    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    device = get_device()
    test_csv = Path("data/splits/test.csv")
    ckpt_path = Path(args.checkpoint)

    if not ckpt_path.exists() or not test_csv.exists():
        print("[ERROR] Checkpoint or test split missing!")
        sys.exit(1)

    test_dataset = DeepfakeDataset(str(test_csv), split='test')

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

    print(f"[ROBUSTNESS] Running transformation benchmark on {ckpt_path.name}...")
    rob_results = evaluate_robustness(model, test_dataset, device=device)

    out_dir = Path("reports/robustness")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{ckpt_path.stem}_robustness_benchmark.json"

    with open(out_file, "w") as f:
        json.dump(rob_results, f, indent=2)

    print(f"[ROBUSTNESS] Completed!")
    print(f"  - Clean ROC-AUC: {rob_results['clean_roc_auc']:.4f}")
    print(f"  - Mean AUC Drop: {rob_results['mean_auc_drop']:.4f}")
    print(f"  - Robustness Score: {rob_results['robustness_score']:.4f}")
    print(f"  - Results saved to {out_file}")

if __name__ == "__main__":
    main()
