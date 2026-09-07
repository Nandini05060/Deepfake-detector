import sys
from pathlib import Path
import yaml

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.split import create_leak_free_splits

def main():
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
        
    audit_csv = Path(config['data']['reports_audit_dir']) / "dataset_audit.csv"
    output_dir = config['data']['splits_dir']
    
    if not audit_csv.exists():
        print(f"[ERROR] Dataset audit CSV not found at {audit_csv}. Run scripts/audit_dataset.py first!")
        sys.exit(1)
        
    create_leak_free_splits(
        str(audit_csv),
        output_dir,
        train_ratio=config['data']['split_ratios']['train'],
        val_ratio=config['data']['split_ratios']['val'],
        test_ratio=config['data']['split_ratios']['test'],
        seed=config['training']['seed']
    )

if __name__ == "__main__":
    main()
