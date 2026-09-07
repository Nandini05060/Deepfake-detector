import sys
import json
import argparse
from pathlib import Path
import yaml
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.simple_cnn import SimpleCNN
from src.models.dense_cnn_baseline import DenseCNNBaseline
from src.models.efficientnet_baseline import EfficientNetBaseline
from src.models.lxfd_model import LXDFDModel
from src.evaluation.cross_domain import evaluate_cross_domain
from src.utils.device import get_device

def main():
    parser = argparse.ArgumentParser(description="Run Protocol B Cross-Domain Generalization Test")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model-type", type=str, default="lxfd", choices=["simple_cnn", "dense_cnn", "efficientnet", "lxfd"])
    parser.add_argument("--fusion", type=str, default="attention")
    parser.add_argument("--in-domain-auc", type=float, default=0.985)
    parser.add_argument("--external-csv", type=str, default="data/external/test.csv")
    args = parser.parse_args()

    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    device = get_device()
    ckpt_path = Path(args.checkpoint)

    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint missing: {ckpt_path}")
        sys.exit(1)

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

    cross_res = evaluate_cross_domain(
        model=model,
        in_domain_auc=args.in_domain_auc,
        external_csv_path=args.external_csv,
        device=device
    )

    out_dir = Path("reports/cross_domain")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{ckpt_path.stem}_cross_domain.json"

    with open(out_file, "w") as f:
        json.dump(cross_res, f, indent=2)

    print(f"[CROSS-DOMAIN] Finished evaluation. Saved to {out_file}")

if __name__ == "__main__":
    main()
