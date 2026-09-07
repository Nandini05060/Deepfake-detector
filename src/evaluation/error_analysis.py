import json
import shutil
from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from PIL import Image

def run_error_analysis(
    model: torch.nn.Module,
    test_dataset,
    device: torch.device,
    output_dir: str = "reports/error_analysis",
    threshold: float = 0.5,
    max_samples_per_category: int = 20
) -> dict:
    """
    Identifies False Positives (FP), False Negatives (FN), True Positives (TP), True Negatives (TN).
    Saves high-confidence mistakes and sample records to reports/error_analysis/.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    fp_dir = out_path / "false_positives"
    fn_dir = out_path / "false_negatives"
    tp_dir = out_path / "true_positives"
    tn_dir = out_path / "true_negatives"
    
    for d in [fp_dir, fn_dir, tp_dir, tn_dir]:
        d.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    model.eval()

    records = []
    
    with torch.no_grad():
        for batch in loader:
            imgs = batch['image'].to(device)
            lbls = batch['label'].cpu().numpy()
            paths = batch['path']
            
            out = model(imgs)
            logits = out[0] if isinstance(out, tuple) else out
            probs = torch.sigmoid(logits).cpu().numpy()

            for i in range(len(lbls)):
                y_true = int(lbls[i]) # 0=REAL, 1=FAKE
                y_prob = float(probs[i])
                y_pred = int(y_prob >= threshold)
                img_path = paths[i]

                if y_true == 0 and y_pred == 1:
                    category = "FP" # Real predicted as Fake
                elif y_true == 1 and y_pred == 0:
                    category = "FN" # Fake predicted as Real
                elif y_true == 1 and y_pred == 1:
                    category = "TP" # Correctly identified Fake
                else:
                    category = "TN" # Correctly identified Real

                # Confidence distance from threshold
                confidence = y_prob if y_pred == 1 else (1.0 - y_prob)

                records.append({
                    "file_path": img_path,
                    "relative_name": Path(img_path).name,
                    "ground_truth": "FAKE" if y_true == 1 else "REAL",
                    "prediction": "FAKE" if y_pred == 1 else "REAL",
                    "fake_probability": round(y_prob, 4),
                    "confidence": round(float(confidence), 4),
                    "category": category
                })

    # Group records
    grouped = {"FP": [], "FN": [], "TP": [], "TN": []}
    for r in records:
        grouped[r["category"]].append(r)

    # Sort mistakes by confidence (highest confidence mistakes first)
    grouped["FP"].sort(key=lambda x: x["confidence"], reverse=True)
    grouped["FN"].sort(key=lambda x: x["confidence"], reverse=True)

    # Copy top samples to error analysis directories
    category_map = {"FP": fp_dir, "FN": fn_dir, "TP": tp_dir, "TN": tn_dir}
    for cat, cat_dir in category_map.items():
        samples = grouped[cat][:max_samples_per_category]
        for idx, item in enumerate(samples):
            src_file = Path(item["file_path"])
            if src_file.exists():
                dst_name = f"{idx+1:02d}_conf{int(item['confidence']*100)}_{src_file.name}"
                dst_path = cat_dir / dst_name
                try:
                    shutil.copy(src_file, dst_path)
                    item["sample_image_saved"] = str(dst_path)
                except Exception:
                    pass

    summary = {
        "total_evaluated": len(records),
        "false_positives_count": len(grouped["FP"]),
        "false_negatives_count": len(grouped["FN"]),
        "true_positives_count": len(grouped["TP"]),
        "true_negatives_count": len(grouped["TN"]),
        "top_false_positives": grouped["FP"][:10],
        "top_false_negatives": grouped["FN"][:10]
    }

    report_path = out_path / "error_analysis_report.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[ERROR ANALYSIS] Identified FP: {len(grouped['FP'])}, FN: {len(grouped['FN'])}. Saved report to {report_path}")
    return summary
