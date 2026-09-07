import sys
import argparse
from pathlib import Path
import numpy as np
import cv2
import torch
from PIL import Image
import torchvision.transforms as T

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.lxfd_model import LXDFDModel
from src.explainability.gradcam import SimpleGradCAM, overlay_heatmap_on_image
from src.frequency.visualization import generate_dct_visualizations
from src.utils.device import get_device

def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM Heatmap & DCT Spectrum for Image")
    parser.add_argument("--image", type=str, required=True, help="Path to input face image")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model .pt file")
    parser.add_argument("--out-dir", type=str, default="reports/figures")
    args = parser.parse_args()

    device = get_device()
    img_path = Path(args.image)
    ckpt_path = Path(args.checkpoint)

    if not img_path.exists() or not ckpt_path.exists():
        print("[ERROR] Input image or checkpoint does not exist!")
        sys.exit(1)

    # Load image
    pil_img = Image.open(img_path).convert('RGB')
    orig_np = np.array(pil_img)
    resized_rgb = cv2.resize(orig_np, (224, 224))

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    tensor_in = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])(pil_img).unsqueeze(0).to(device)

    # Load model
    model = LXDFDModel(fusion_type='attention', pretrained=False)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # Compute prediction probability
    with torch.no_grad():
        out = model(tensor_in)
        logit = out[0] if isinstance(out, tuple) else out
        prob_fake = torch.sigmoid(logit).item()

    pred_label = "FAKE" if prob_fake >= 0.5 else "REAL"

    # Grad-CAM
    cam_generator = SimpleGradCAM(model)
    heatmap = cam_generator.generate_heatmap(tensor_in, target_class=1)
    overlay = overlay_heatmap_on_image(resized_rgb, heatmap, alpha=0.5)

    # DCT Visualizations
    dct_vis = generate_dct_visualizations(resized_rgb)

    # Save outputs
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = img_path.stem

    cv2.imwrite(str(out_dir / f"{base_name}_original.jpg"), cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / f"{base_name}_heatmap.jpg"), cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR))
    cv2.imwrite(str(out_dir / f"{base_name}_overlay.jpg"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / f"{base_name}_dct_spectrum.jpg"), cv2.cvtColor(dct_vis["dct_colormap"], cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / f"{base_name}_dct_high_freq.jpg"), cv2.cvtColor(dct_vis["high_freq_colormap"], cv2.COLOR_RGB2BGR))

    print(f"[EXPLAINABILITY] Image Analysis Complete for {img_path.name}:")
    print(f"  - Prediction: {pred_label} (Fake Probability: {prob_fake*100:.2f}%)")
    print(f"  - Wording: Highlighted regions indicate image areas that contributed strongly to the model's prediction.")
    print(f"  - Saved figures to {out_dir}")

if __name__ == "__main__":
    main()
