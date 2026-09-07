import os
import sys
import yaml
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset_auditor import run_full_dataset_audit

def main():
    config_path = Path("configs/config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    dataset_dir = r"C:\Users\nandi\Desktop\DEEPFAKE DETECTION\Final Dataset"
    audit_dir = config['data']['reports_audit_dir']
    processed_dir = config['data']['processed_dir']
    splits_dir = config['data']['splits_dir']
    metadata_path = "data/metadata.csv"
    yunet_model = "yunet.onnx"
    
    run_full_dataset_audit(
        dataset_dir=dataset_dir,
        output_audit_dir=audit_dir,
        output_processed_dir=processed_dir,
        output_splits_dir=splits_dir,
        output_metadata_path=metadata_path,
        yunet_model_path=yunet_model,
        phash_threshold=6
    )

if __name__ == "__main__":
    main()
