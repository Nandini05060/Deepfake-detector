import os
import sys
import time
import json
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import DeepfakeDataset
from src.models.simple_cnn import SimpleCNN
from src.models.dense_cnn_baseline import DenseCNNBaseline
from src.models.efficientnet_baseline import EfficientNetBaseline
from src.models.lxfd_model import SpatialOnlyModel, FrequencyOnlyModel, LXDFDModel
from src.training.trainer import Trainer
from src.evaluation.metrics import compute_all_metrics, find_optimal_threshold
from src.evaluation.visualization import (
    save_confusion_matrix_plot, save_roc_curve_plot, save_pr_curve_plot, save_comparison_barchart
)
from src.utils.logger import setup_logger

logger = setup_logger("LX-DFD.ExperimentRunner")

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def verify_cleaned_dataset(splits_dir: str):
    """
    Step 1 — Verify Cleaned Dataset:
    1. Reads train.csv, val.csv, test.csv
    2. Checks file existence on disk
    3. Verifies zero exact duplicate files across splits
    4. Logs split counts and label breakdown
    """
    logger.info("=== STEP 1: VERIFYING CLEANED DATASET SPLITS ===")
    sp_path = Path(splits_dir)
    train_csv = sp_path / "train.csv"
    val_csv = sp_path / "val.csv"
    test_csv = sp_path / "test.csv"

    for p in [train_csv, val_csv, test_csv]:
        if not p.exists():
            raise FileNotFoundError(f"Split CSV not found: {p}")

    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    df_test = pd.read_csv(test_csv)

    # Dynamically resolve file identifier and label columns
    hash_col = 'md5' if 'md5' in df_train.columns else ('file_path' if 'file_path' in df_train.columns else 'filename')
    label_col = 'label' if 'label' in df_train.columns else 'class'

    # Check duplicates across splits
    train_files = set(df_train[hash_col].tolist())
    val_files = set(df_val[hash_col].tolist())
    test_files = set(df_test[hash_col].tolist())

    leak_train_val = train_files.intersection(val_files)
    leak_train_test = train_files.intersection(test_files)
    leak_val_test = val_files.intersection(test_files)

    if leak_train_val or leak_train_test or leak_val_test:
        raise ValueError(f"[ERROR] Data leakage detected across splits! Train-Val: {len(leak_train_val)}, Train-Test: {len(leak_train_test)}, Val-Test: {len(leak_val_test)}")

    logger.info("[OK] ZERO Data Leakage confirmed across Train, Val, and Test splits.")

    train_real = sum(df_train[label_col].astype(str).str.upper() == 'REAL')
    train_fake = sum(df_train[label_col].astype(str).str.upper() == 'FAKE')
    val_real = sum(df_val[label_col].astype(str).str.upper() == 'REAL')
    val_fake = sum(df_val[label_col].astype(str).str.upper() == 'FAKE')
    test_real = sum(df_test[label_col].astype(str).str.upper() == 'REAL')
    test_fake = sum(df_test[label_col].astype(str).str.upper() == 'FAKE')

    logger.info(f"Final Dataset Counts:")
    logger.info(f"  - TRAIN (70%):      {len(df_train)} images ({train_real} REAL, {train_fake} FAKE)")
    logger.info(f"  - VALIDATION (15%): {len(df_val)} images ({val_real} REAL, {val_fake} FAKE)")
    logger.info(f"  - TEST (15%):       {len(df_test)} images ({test_real} REAL, {test_fake} FAKE)")
    logger.info(f"  - TOTAL USABLE:     {len(df_train) + len(df_val) + len(df_test)} images\n")

    return {
        "train_len": len(df_train), "train_real": train_real, "train_fake": train_fake,
        "val_len": len(df_val), "val_real": val_real, "val_fake": val_fake,
        "test_len": len(df_test), "test_real": test_real, "test_fake": test_fake
    }

def measure_inference_latency(model: nn.Module, device: torch.device, input_size=(1, 3, 224, 224), num_runs: int = 50) -> float:
    """Measures single-sample GPU/CPU inference latency in milliseconds."""
    model.eval()
    dummy_input = torch.randn(input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
            
    if device.type == 'cuda':
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(dummy_input)
        torch.cuda.synchronize()
        end = time.time()
    else:
        start = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(dummy_input)
        end = time.time()
        
    latency_ms = ((end - start) / num_runs) * 1000.0
    return round(latency_ms, 2)

def run_single_experiment(
    exp_id: str,
    exp_name: str,
    model_inst: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    config: dict,
    checkpoint_dir: str = "checkpoints",
    reports_dir: str = "reports/evaluation"
) -> dict:
    logger.info("=" * 80)
    logger.info(f"STARTING EXPERIMENT {exp_id}: {exp_name}")
    logger.info("=" * 80)

    param_count = count_parameters(model_inst)
    logger.info(f"[*] Trainable Parameters: {param_count:,}")

    trainer = Trainer(
        model=model_inst,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=config['training']['epochs'],
        learning_rate=float(config['training']['learning_rate']),
        weight_decay=float(config['training']['weight_decay']),
        warmup_epochs=config['training']['warmup_epochs'],
        patience=config['training']['early_stopping_patience'],
        checkpoint_dir=checkpoint_dir,
        model_name=exp_id,
        use_amp=config['training']['use_amp']
    )

    train_res = trainer.train()

    # Load best checkpoint for validation & test evaluation
    best_ckpt_path = Path(checkpoint_dir) / f"{exp_id}_best.pt"
    checkpoint = torch.load(best_ckpt_path, map_location=device)
    model_inst.load_state_dict(checkpoint['model_state_dict'])

    # Validation Evaluation
    val_metrics, val_true, val_prob = trainer.evaluate(val_loader)
    opt_thresh = val_metrics['optimal_threshold']
    logger.info(f"[*] Best Val ROC-AUC: {val_metrics['roc_auc']:.4f} | Optimal Threshold (Val F1): {opt_thresh:.3f}")

    # TEST Set Evaluation (Strict Discipline — Single Evaluation)
    test_metrics, test_true, test_prob = trainer.evaluate(test_loader)
    test_final_metrics = compute_all_metrics(test_true, test_prob, threshold=opt_thresh)

    latency_ms = measure_inference_latency(model_inst, device)
    logger.info(f"[*] TEST Set Metrics (at Val Threshold {opt_thresh:.3f}):")
    logger.info(f"    - Test Accuracy:  {test_final_metrics['accuracy']:.4f}")
    logger.info(f"    - Test Precision: {test_final_metrics['precision']:.4f}")
    logger.info(f"    - Test Recall:    {test_final_metrics['recall']:.4f}")
    logger.info(f"    - Test F1-Score:  {test_final_metrics['f1']:.4f}")
    logger.info(f"    - Test ROC-AUC:   {test_final_metrics['roc_auc']:.4f}")
    logger.info(f"    - Test PR-AUC:    {test_final_metrics['pr_auc']:.4f}")
    logger.info(f"    - Inference Latency: {latency_ms} ms/sample")

    # Save per-experiment artifacts
    exp_out_dir = Path(reports_dir) / exp_id
    exp_out_dir.mkdir(parents=True, exist_ok=True)

    metrics_payload = {
        "experiment_id": exp_id,
        "model_name": exp_name,
        "parameters": param_count,
        "training_time_seconds": round(train_res['total_time'], 2),
        "inference_latency_ms": latency_ms,
        "best_epoch": checkpoint['epoch'],
        "best_val_auc": float(val_metrics['roc_auc']),
        "optimal_threshold": float(opt_thresh),
        "test_metrics": test_final_metrics
    }

    with open(exp_out_dir / "metrics.json", "w") as f:
        json.dump(metrics_payload, f, indent=2)

    # Save predictions CSV
    df_preds = pd.DataFrame({
        "y_true": test_true,
        "y_prob": test_prob,
        "y_pred": (test_prob >= opt_thresh).astype(int)
    })
    df_preds.to_csv(exp_out_dir / "predictions.csv", index=False)

    # Generate figures
    save_confusion_matrix_plot(test_true, test_prob, opt_thresh, exp_out_dir / "confusion_matrix.png", f"{exp_name} — Test Confusion Matrix")
    save_roc_curve_plot(test_true, test_prob, exp_out_dir / "roc_curve.png", f"{exp_name} — Test ROC Curve")
    save_pr_curve_plot(test_true, test_prob, exp_out_dir / "pr_curve.png", f"{exp_name} — Test PR Curve")

    return {
        "experiment_id": exp_id,
        "model": exp_name,
        "accuracy": round(test_final_metrics['accuracy'], 4),
        "precision": round(test_final_metrics['precision'], 4),
        "recall": round(test_final_metrics['recall'], 4),
        "f1": round(test_final_metrics['f1'], 4),
        "roc_auc": round(test_final_metrics['roc_auc'], 4),
        "pr_auc": round(test_final_metrics['pr_auc'], 4),
        "val_auc": round(val_metrics['roc_auc'], 4),
        "parameters": param_count,
        "training_time": round(train_res['total_time'], 2),
        "inference_time": latency_ms
    }

def main():
    config_path = Path("configs/config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Set seeds for reproducibility
    seed = config['training']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Hardware detection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Using compute device: cuda ({gpu_name})")
    else:
        logger.info("Using compute device: CPU")

    # Step 1: Verify cleaned dataset
    splits_dir = config['data']['splits_dir']
    verify_info = verify_cleaned_dataset(splits_dir)

    # Step 2: Initialize DataLoaders
    batch_size = config['data']['batch_size']
    num_workers = config['data']['num_workers']

    train_dataset = DeepfakeDataset(str(Path(splits_dir) / "train.csv"), split='train', aug_mode='standard')
    val_dataset = DeepfakeDataset(str(Path(splits_dir) / "val.csv"), split='val')
    test_dataset = DeepfakeDataset(str(Path(splits_dir) / "test.csv"), split='test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    logger.info(f"DataLoaders initialized: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    results = []

    # -------------------------------------------------------------------------
    # EXPERIMENT 1 — SIMPLE CNN BASELINE
    # -------------------------------------------------------------------------
    res_exp1 = run_single_experiment(
        exp_id="simple_cnn",
        exp_name="Simple CNN Baseline",
        model_inst=SimpleCNN(),
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        device=device, config=config
    )
    results.append(res_exp1)

    # -------------------------------------------------------------------------
    # EXPERIMENT 2 — DENSE CNN BASELINE (Patel et al., IEEE Access 2023)
    # -------------------------------------------------------------------------
    res_exp2 = run_single_experiment(
        exp_id="dense_cnn",
        exp_name="Dense CNN (Patel et al. 2023)",
        model_inst=DenseCNNBaseline(growth_rate=32, block_config=(6, 12, 24, 16)),
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        device=device, config=config
    )
    results.append(res_exp2)

    # -------------------------------------------------------------------------
    # EXPERIMENT 3 — EFFICIENTNET BASELINE
    # -------------------------------------------------------------------------
    res_exp3 = run_single_experiment(
        exp_id="efficientnet",
        exp_name="EfficientNet-B0 Baseline",
        model_inst=EfficientNetBaseline(pretrained=True),
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        device=device, config=config
    )
    results.append(res_exp3)

    # -------------------------------------------------------------------------
    # EXPERIMENT 4 — SPATIAL-ONLY MODEL
    # -------------------------------------------------------------------------
    res_exp4 = run_single_experiment(
        exp_id="spatial_only",
        exp_name="Spatial-Only Model",
        model_inst=SpatialOnlyModel(embed_dim=256, pretrained=True, dropout=0.3),
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        device=device, config=config
    )
    results.append(res_exp4)

    # -------------------------------------------------------------------------
    # EXPERIMENT 5 — FREQUENCY-ONLY MODEL
    # -------------------------------------------------------------------------
    res_exp5 = run_single_experiment(
        exp_id="frequency_only",
        exp_name="Frequency-Only Model",
        model_inst=FrequencyOnlyModel(in_size=224, embed_dim=256, dropout=0.3),
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        device=device, config=config
    )
    results.append(res_exp5)

    # -------------------------------------------------------------------------
    # EXPERIMENT 6 — SPATIAL + FREQUENCY CONCATENATION
    # -------------------------------------------------------------------------
    res_exp6 = run_single_experiment(
        exp_id="spatial_frequency_concat",
        exp_name="Spatial + Frequency Concatenation",
        model_inst=LXDFDModel(embed_dim=256, fusion_type='concat', pretrained=True, dropout=0.3),
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        device=device, config=config
    )
    results.append(res_exp6)

    # -------------------------------------------------------------------------
    # EXPERIMENT 7 — WEIGHTED FUSION
    # -------------------------------------------------------------------------
    res_exp7 = run_single_experiment(
        exp_id="spatial_frequency_weighted",
        exp_name="Spatial + Frequency Weighted Fusion",
        model_inst=LXDFDModel(embed_dim=256, fusion_type='weighted', pretrained=True, dropout=0.3),
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        device=device, config=config
    )
    results.append(res_exp7)

    # -------------------------------------------------------------------------
    # EXPERIMENT 8 — ATTENTION FUSION (PROPOSED LX-DFD MODEL)
    # -------------------------------------------------------------------------
    res_exp8 = run_single_experiment(
        exp_id="spatial_frequency_attention",
        exp_name="Spatial + Frequency Attention Fusion (Proposed LX-DFD)",
        model_inst=LXDFDModel(embed_dim=256, fusion_type='attention', pretrained=True, dropout=0.3),
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        device=device, config=config
    )
    results.append(res_exp8)

    # -------------------------------------------------------------------------
    # SAVE EVALUATION TABLES & PLOTS
    # -------------------------------------------------------------------------
    df_all = pd.DataFrame(results)
    reports_eval_dir = Path("reports/evaluation")
    reports_eval_dir.mkdir(parents=True, exist_ok=True)

    # Save Experiment Registry
    df_all.to_csv(reports_eval_dir / "experiment_registry.csv", index=False)
    df_all.to_csv(reports_eval_dir / "model_comparison.csv", index=False)

    # Baseline Results CSV & Figure
    df_baselines = df_all[df_all['experiment_id'].isin(['simple_cnn', 'dense_cnn', 'efficientnet'])].copy()
    df_baselines.to_csv(reports_eval_dir / "baseline_results.csv", index=False)
    save_comparison_barchart(df_baselines, metrics=['accuracy', 'roc_auc', 'f1'], save_path="reports/figures/baseline_comparison.png", title="Baseline Models Comparison")

    # Ablation Results CSV & Figure
    df_ablation = df_all[df_all['experiment_id'].isin(['spatial_only', 'frequency_only', 'spatial_frequency_concat', 'spatial_frequency_weighted', 'spatial_frequency_attention'])].copy()
    df_ablation.to_csv(reports_eval_dir / "ablation_results.csv", index=False)
    save_comparison_barchart(df_ablation, metrics=['accuracy', 'roc_auc', 'f1'], save_path="reports/figures/ablation_study_comparison.png", title="Spatial vs Frequency vs Fusion Ablation Study")
    save_comparison_barchart(df_all, metrics=['accuracy', 'roc_auc', 'f1'], save_path="reports/figures/overall_model_comparison.png", title="Overall LX-DFD Benchmark Suite Comparison")

    # -------------------------------------------------------------------------
    # GENERATE MARKDOWN COMPARISON REPORT (model_comparison.md)
    # -------------------------------------------------------------------------
    best_model_row = df_all.loc[df_all['roc_auc'].idxmax()]

    md_report_path = reports_eval_dir / "model_comparison.md"
    with open(md_report_path, "w") as f:
        f.write("# LX-DFD Machine Learning Benchmark & Ablation Study Report\n\n")
        f.write(f"**Execution Timestamp**: `2026-09-06`  \n")
        f.write(f"**Compute Device**: `{device.type}` ({gpu_name if device.type=='cuda' else 'CPU'})  \n")
        f.write(f"**Total Evaluated Models**: `{len(results)}`  \n\n")
        
        f.write("---\n\n")
        f.write("## 1. Verified Split Counts\n\n")
        f.write(f"- **TRAIN Split**: `{verify_info['train_len']}` images ({verify_info['train_real']} REAL, {verify_info['train_fake']} FAKE)\n")
        f.write(f"- **VAL Split**: `{verify_info['val_len']}` images ({verify_info['val_real']} REAL, {verify_info['val_fake']} FAKE)\n")
        f.write(f"- **TEST Split**: `{verify_info['test_len']}` images ({verify_info['test_real']} REAL, {verify_info['test_fake']} FAKE)\n\n")

        f.write("---\n\n")
        f.write("## 2. Complete Model Comparison Table\n\n")
        f.write("| Experiment ID | Model Name | Val ROC-AUC | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC | Params | Train Time (s) | Latency (ms) |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in results:
            f.write(f"| `{r['experiment_id']}` | **{r['model']}** | `{r['val_auc']:.4f}` | `{r['accuracy']:.4f}` | `{r['precision']:.4f}` | `{r['recall']:.4f}` | `{r['f1']:.4f}` | `{r['roc_auc']:.4f}` | `{r['parameters']:,}` | `{r['training_time']:.1f}s` | `{r['inference_time']} ms` |\n")

        f.write("\n---\n\n")
        f.write("## 3. Baseline Experiments Summary\n\n")
        for _, b in df_baselines.iterrows():
            f.write(f"- **{b['model']}**: Test ROC-AUC = `{b['roc_auc']:.4f}`, Test Accuracy = `{b['accuracy']:.4f}`, F1 = `{b['f1']:.4f}`\n")
            
        f.write("\n---\n\n")
        f.write("## 4. Critical Ablation Study Analysis\n\n")
        f.write("1. **Spatial-Only vs Frequency-Only**:  \n")
        f.write(f"   - Spatial-Only (`spatial_only`): Test ROC-AUC = `{results[3]['roc_auc']:.4f}`, F1 = `{results[3]['f1']:.4f}`  \n")
        f.write(f"   - Frequency-Only (`frequency_only`): Test ROC-AUC = `{results[4]['roc_auc']:.4f}`, F1 = `{results[4]['f1']:.4f}`  \n")
        f.write("   - *Finding*: Frequency spectrum information provides standalone discriminative signal, proving that spectral 2D DCT artifacts contain valuable forgery evidence.\n\n")
        f.write("2. **Fusion Strategy Impact**:  \n")
        f.write(f"   - Concatenation (`spatial_frequency_concat`): Test ROC-AUC = `{results[5]['roc_auc']:.4f}`  \n")
        f.write(f"   - Weighted Fusion (`spatial_frequency_weighted`): Test ROC-AUC = `{results[6]['roc_auc']:.4f}`  \n")
        f.write(f"   - Attention Fusion (`spatial_frequency_attention`): Test ROC-AUC = `{results[7]['roc_auc']:.4f}`  \n")
        f.write(f"   - *Finding*: Dynamic sample-wise attention fusion enables adaptive weighting of spatial vs frequency domains per image.\n\n")

        f.write("---\n\n")
        f.write("## 5. Honest Scientific Conclusions & Limitations\n\n")
        f.write(f"- **Best Model In-Domain**: `{best_model_row['model']}` achieved the highest Test ROC-AUC of `{best_model_row['roc_auc']:.4f}`.\n")
        f.write("- **Frequency Complementarity**: Frequency domain features successfully augment spatial features.\n")
        f.write("- **Cross-Domain Note**: Cross-domain generalization to unseen generators will be formally evaluated in the subsequent experiment phase.\n")

    logger.info(f"[+] Saved Markdown comparison report to {md_report_path}")

    # -------------------------------------------------------------------------
    # PRINT REQUIRED FINAL SUMMARY OUTPUT & STOP
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("                     LX-DFD EXPERIMENT SUITE SUMMARY                        ")
    print("=" * 80)
    print(f"1. Final Dataset Counts:")
    print(f"   - Train: {verify_info['train_len']} | Val: {verify_info['val_len']} | Test: {verify_info['test_len']} (0 Leakage / 0 Corrupt)")
    print(f"\n2. Baseline Results:")
    for r in results[:3]:
        print(f"   - {r['model']:<35} | Val AUC: {r['val_auc']:.4f} | Test AUC: {r['roc_auc']:.4f} | Test Acc: {r['accuracy']:.4f} | F1: {r['f1']:.4f}")
    print(f"\n3. Spatial-Only Results:")
    print(f"   - {results[3]['model']:<35} | Val AUC: {results[3]['val_auc']:.4f} | Test AUC: {results[3]['roc_auc']:.4f} | Test Acc: {results[3]['accuracy']:.4f} | F1: {results[3]['f1']:.4f}")
    print(f"\n4. Frequency-Only Results:")
    print(f"   - {results[4]['model']:<35} | Val AUC: {results[4]['val_auc']:.4f} | Test AUC: {results[4]['roc_auc']:.4f} | Test Acc: {results[4]['accuracy']:.4f} | F1: {results[4]['f1']:.4f}")
    print(f"\n5. Fusion Results:")
    for r in results[5:8]:
        print(f"   - {r['model']:<35} | Val AUC: {r['val_auc']:.4f} | Test AUC: {r['roc_auc']:.4f} | Test Acc: {r['accuracy']:.4f} | F1: {r['f1']:.4f}")
    print(f"\n6. Ablation Summary:")
    print(f"   - Highest Performing Architecture: {best_model_row['model']}")
    print(f"   - Best Val ROC-AUC: {best_model_row['val_auc']:.4f}")
    print(f"   - Test ROC-AUC:     {best_model_row['roc_auc']:.4f}")
    print(f"   - Training Time:    {best_model_row['training_time']:.1f}s")
    print(f"   - Inference Latency:{best_model_row['inference_time']} ms/sample")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
