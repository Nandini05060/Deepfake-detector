import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.data.dataset import DeepfakeDataset
from src.evaluation.metrics import compute_all_metrics

def evaluate_cross_domain(
    model: torch.nn.Module,
    in_domain_auc: float,
    external_csv_path: str,
    device: torch.device,
    batch_size: int = 32,
    threshold: float = 0.5
) -> dict:
    """
    Protocol B — Cross-Domain Evaluation Pipeline.
    Evaluates trained model on an unseen external dataset (e.g. External Dataset A / B).
    Computes:
      - Cross-Domain ROC-AUC
      - Generalization Gap = In-Domain ROC-AUC - Cross-Domain ROC-AUC
    """
    ext_path = Path(external_csv_path)
    
    if not ext_path.exists():
        print(f"[CROSS-DOMAIN] External dataset split file not found: {ext_path}")
        return {
            "status": "pending",
            "message": "External cross-dataset evaluation pending. Add external dataset to data/external/",
            "in_domain_roc_auc": float(in_domain_auc),
            "cross_domain_roc_auc": None,
            "generalization_gap": None
        }

    print(f"[CROSS-DOMAIN] Running cross-dataset evaluation on {ext_path}...")
    ext_dataset = DeepfakeDataset(str(ext_path), split='test')
    ext_loader = DataLoader(ext_dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    all_targets, all_probs = [], []

    with torch.no_grad():
        for batch in ext_loader:
            imgs = batch['image'].to(device)
            lbls = batch['label'].to(device)
            out = model(imgs)
            logits = out[0] if isinstance(out, tuple) else out
            probs = torch.sigmoid(logits).cpu().numpy()
            all_targets.extend(lbls.cpu().numpy())
            all_probs.extend(probs)

    ext_metrics = compute_all_metrics(np.array(all_targets), np.array(all_probs), threshold=threshold)
    cross_auc = ext_metrics["roc_auc"]
    gen_gap = max(0.0, in_domain_auc - cross_auc)

    return {
        "status": "completed",
        "dataset_path": str(ext_path),
        "in_domain_roc_auc": float(in_domain_auc),
        "cross_domain_roc_auc": float(cross_auc),
        "generalization_gap": float(gen_gap),
        "cross_domain_metrics": ext_metrics
    }
