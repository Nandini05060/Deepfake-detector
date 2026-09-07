import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc

def save_confusion_matrix_plot(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, save_path: str, title: str = "Confusion Matrix"):
    """Saves formatted confusion matrix heatmap plot."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['REAL (0)', 'FAKE (1)'],
                yticklabels=['REAL (0)', 'FAKE (1)'],
                annot_kws={"size": 14, "weight": "bold"})
    plt.title(title, fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Predicted Label', fontsize=11, fontweight='bold')
    plt.ylabel('True Label', fontsize=11, fontweight='bold')
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def save_roc_curve_plot(y_true: np.ndarray, y_prob: np.ndarray, save_path: str, title: str = "ROC Curve"):
    """Saves ROC Curve plot."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='#2563eb', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='#94a3b8', lw=1.5, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=11, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=11, fontweight='bold')
    plt.title(title, fontsize=12, fontweight='bold', pad=12)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def save_pr_curve_plot(y_true: np.ndarray, y_prob: np.ndarray, save_path: str, title: str = "Precision-Recall Curve"):
    """Saves Precision-Recall Curve plot."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color='#059669', lw=2.5, label=f'PR Curve (AUC = {pr_auc:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=11, fontweight='bold')
    plt.ylabel('Precision', fontsize=11, fontweight='bold')
    plt.title(title, fontsize=12, fontweight='bold', pad=12)
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def save_comparison_barchart(df_results, metrics=['accuracy', 'roc_auc', 'f1'], save_path: str = "reports/figures/model_comparison.png", title: str = "Model Performance Comparison"):
    """Saves bar chart comparing multiple models across metrics."""
    plt.figure(figsize=(10, 6))
    
    models = df_results['model'].tolist()
    x = np.arange(len(models))
    width = 0.25
    
    colors = ['#2563eb', '#059669', '#7c3aed', '#d97706']
    
    for i, metric in enumerate(metrics):
        if metric in df_results.columns:
            vals = df_results[metric].values
            plt.bar(x + (i - 1) * width, vals, width, label=metric.upper().replace('_', '-'), color=colors[i % len(colors)])
            
            # Annotate bar values
            for xi, val in zip(x + (i - 1) * width, vals):
                plt.text(xi, val + 0.01, f"{val:.3f}", ha='center', va='bottom', fontsize=8, rotation=90)
                
    plt.xlabel('Model Architecture', fontsize=11, fontweight='bold')
    plt.ylabel('Score', fontsize=11, fontweight='bold')
    plt.title(title, fontsize=13, fontweight='bold', pad=15)
    plt.xticks(x, models, rotation=15, ha='right', fontsize=10)
    plt.ylim([0.0, 1.15])
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
