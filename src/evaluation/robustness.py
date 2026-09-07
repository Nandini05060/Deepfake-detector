import io
import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from src.evaluation.metrics import compute_all_metrics

class RobustnessDatasetWrapper(Dataset):
    """
    Wraps a base dataset and applies a controlled transformation perturbation on-the-fly.
    """
    def __init__(self, base_dataset, transform_fn):
        self.base_dataset = base_dataset
        self.transform_fn = transform_fn

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # We fetch original image path and label from base dataset row
        row = self.base_dataset.df.iloc[idx]
        img_path = row['file_path']
        label_val = self.base_dataset.label_map.get(str(row['label']).upper(), 0.0)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224), color=(0, 0, 0))
            
        transformed_img = self.transform_fn(image)
        
        # Standard normalization
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        tensor_img = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])(transformed_img)
        
        return {
            'image': tensor_img,
            'label': torch.tensor(label_val, dtype=torch.float32),
            'path': img_path
        }

def get_perturbation_transforms() -> dict:
    """
    Returns a dictionary of controlled transformation functions for robustness benchmarking.
    """
    transforms_dict = {}

    # 1. JPEG Compression
    for q in [95, 85, 75, 60, 40]:
        def make_jpeg_fn(quality):
            def jpeg_fn(img: Image.Image) -> Image.Image:
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=quality)
                buf.seek(0)
                return Image.open(buf).convert("RGB")
            return jpeg_fn
        transforms_dict[f"jpeg_q{q}"] = make_jpeg_fn(q)

    # 2. Gaussian Blur
    for sigma in [0.5, 1.0, 2.0]:
        def make_blur_fn(s):
            def blur_fn(img: Image.Image) -> Image.Image:
                return img.filter(ImageFilter.GaussianBlur(radius=s))
            return blur_fn
        transforms_dict[f"blur_sigma_{sigma}"] = make_blur_fn(sigma)

    # 3. Resize Degradation
    for pct in [75, 50, 25]:
        def make_resize_fn(p_val):
            def resize_fn(img: Image.Image) -> Image.Image:
                w, h = img.size
                nw, nh = max(1, int(w * p_val / 100.0)), max(1, int(h * p_val / 100.0))
                down = img.resize((nw, nh), Image.Resampling.BILINEAR)
                return down.resize((w, h), Image.Resampling.BILINEAR)
            return resize_fn
        transforms_dict[f"resize_{pct}pct"] = make_resize_fn(pct)

    # 4. Color Jitter (Brightness / Contrast)
    def brightness_fn(img: Image.Image) -> Image.Image:
        return ImageEnhance.Brightness(img).enhance(1.3)
    transforms_dict["color_brightness_130"] = brightness_fn

    def contrast_fn(img: Image.Image) -> Image.Image:
        return ImageEnhance.Contrast(img).enhance(1.3)
    transforms_dict["color_contrast_130"] = contrast_fn

    # 5. Crop
    for crop_pct in [5, 10, 20]:
        def make_crop_fn(c_pct):
            def crop_fn(img: Image.Image) -> Image.Image:
                w, h = img.size
                crop_w = int(w * (1.0 - c_pct / 100.0))
                crop_h = int(h * (1.0 - c_pct / 100.0))
                left = (w - crop_w) // 2
                top = (h - crop_h) // 2
                cropped = img.crop((left, top, left + crop_w, top + crop_h))
                return cropped.resize((w, h), Image.Resampling.BILINEAR)
            return crop_fn
        transforms_dict[f"crop_{crop_pct}pct"] = make_crop_fn(crop_pct)

    return transforms_dict

def evaluate_robustness(
    model: torch.nn.Module,
    test_dataset,
    device: torch.device,
    batch_size: int = 32,
    threshold: float = 0.5
) -> dict:
    """
    Runs the controlled Transformation Robustness Benchmark on held-out test data.
    Computes performance under clean vs perturbed conditions and calculates Robustness Score.
    """
    model.eval()
    perturbations = get_perturbation_transforms()
    results = {}

    clean_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 1. Clean Baseline Evaluation
    clean_targets, clean_probs = [], []
    with torch.no_grad():
        for batch in clean_loader:
            imgs = batch['image'].to(device)
            lbls = batch['label'].to(device)
            out = model(imgs)
            logits = out[0] if isinstance(out, tuple) else out
            probs = torch.sigmoid(logits).cpu().numpy()
            clean_targets.extend(lbls.cpu().numpy())
            clean_probs.extend(probs)
            
    clean_metrics = compute_all_metrics(np.array(clean_targets), np.array(clean_probs), threshold=threshold)
    results["clean"] = clean_metrics
    clean_auc = clean_metrics["roc_auc"]

    # 2. Perturbation Evaluations
    auc_drops = []
    for name, transform_fn in perturbations.items():
        rob_dataset = RobustnessDatasetWrapper(test_dataset, transform_fn)
        rob_loader = DataLoader(rob_dataset, batch_size=batch_size, shuffle=False)
        
        targets, probs = [], []
        with torch.no_grad():
            for batch in rob_loader:
                imgs = batch['image'].to(device)
                lbls = batch['label'].to(device)
                out = model(imgs)
                logits = out[0] if isinstance(out, tuple) else out
                p = torch.sigmoid(logits).cpu().numpy()
                targets.extend(lbls.cpu().numpy())
                probs.extend(p)
                
        p_metrics = compute_all_metrics(np.array(targets), np.array(probs), threshold=threshold)
        auc_drop = max(0.0, clean_auc - p_metrics["roc_auc"])
        p_metrics["auc_drop"] = float(auc_drop)
        results[name] = p_metrics
        auc_drops.append(auc_drop)

    mean_auc_drop = float(np.mean(auc_drops)) if auc_drops else 0.0
    robustness_score = float(max(0.0, 1.0 - mean_auc_drop))

    return {
        "clean_roc_auc": float(clean_auc),
        "mean_auc_drop": float(mean_auc_drop),
        "robustness_score": float(robustness_score),
        "perturbation_results": results
    }
