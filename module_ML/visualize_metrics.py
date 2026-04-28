"""Generate visualization untuk metrik evaluasi model baseline dan transformer."""

import json
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

from config import REPORT_DIR, MODEL_DIR

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 5)


def load_metrics(report_path: Path) -> Dict[str, Any]:
    """Load metrics dari file JSON."""
    with report_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def plot_metrics_comparison():
    """Visualize perbandingan metrik baseline vs transformer."""
    baseline_path = REPORT_DIR / "baseline_logreg_metrics.json"
    transformer_path = REPORT_DIR / "transformer_metrics.json"
    
    if not baseline_path.exists() or not transformer_path.exists():
        print(f"⚠️  Laporan belum tersedia. Jalankan train_run.py terlebih dahulu.")
        return
    
    baseline = load_metrics(baseline_path)
    transformer = load_metrics(transformer_path)
    
    metrics_names = ["accuracy", "macro_f1", "weighted_f1"]
    baseline_scores = [baseline[m] for m in metrics_names]
    transformer_scores = [transformer["metrics"].get(f"eval_{m}", 0) for m in metrics_names]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, baseline_scores, width, label="Baseline (TF-IDF + LogReg)", color="#3498db", alpha=0.8)
    bars2 = ax.bar(x + width/2, transformer_scores, width, label="Transformer (IndoBERT)", color="#e74c3c", alpha=0.8)
    
    ax.set_xlabel("Metrik Evaluasi", fontsize=12, fontweight='bold')
    ax.set_ylabel("Skor", fontsize=12, fontweight='bold')
    ax.set_title("Perbandingan Performa: Baseline vs Transformer", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics_names])
    ax.legend(fontsize=10)
    ax.set_ylim([0, 1.0])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    fig.savefig(REPORT_DIR / "metrics_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✓ Visualisasi metrics comparison tersimpan: {REPORT_DIR}/metrics_comparison.png")
    plt.close()


def plot_confusion_matrices():
    """Visualize confusion matrix dari baseline dan transformer."""
    baseline_path = REPORT_DIR / "baseline_logreg_metrics.json"
    transformer_path = REPORT_DIR / "transformer_metrics.json"
    
    if not baseline_path.exists() or not transformer_path.exists():
        print(f"⚠️  Laporan belum tersedia.")
        return
    
    baseline = load_metrics(baseline_path)
    transformer = load_metrics(transformer_path)
    
    class_labels = ["Negatif", "Netral", "Positif"]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Baseline confusion matrix
    if "confusion_matrix" in baseline:
        cm_baseline = np.array(baseline["confusion_matrix"])
        sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='Blues', ax=ax1,
                    xticklabels=class_labels, yticklabels=class_labels, cbar=True)
        ax1.set_title("Confusion Matrix: Baseline (TF-IDF + LogReg)", fontsize=12, fontweight='bold')
        ax1.set_ylabel("Actual", fontsize=11)
        ax1.set_xlabel("Predicted", fontsize=11)
    
    # Transformer confusion matrix
    if "metrics" in transformer and "confusion_matrix" in transformer["metrics"]:
        cm_transformer = np.array(transformer["metrics"]["confusion_matrix"])
        sns.heatmap(cm_transformer, annot=True, fmt='d', cmap='Reds', ax=ax2,
                    xticklabels=class_labels, yticklabels=class_labels, cbar=True)
        ax2.set_title("Confusion Matrix: Transformer (IndoBERT)", fontsize=12, fontweight='bold')
        ax2.set_ylabel("Actual", fontsize=11)
        ax2.set_xlabel("Predicted", fontsize=11)
    
    plt.tight_layout()
    fig.savefig(REPORT_DIR / "confusion_matrices.png", dpi=300, bbox_inches='tight')
    print(f"✓ Visualisasi confusion matrices tersimpan: {REPORT_DIR}/confusion_matrices.png")
    plt.close()


def plot_model_summary():
    """Create summary card dengan info model dan key metrics."""
    baseline_path = REPORT_DIR / "baseline_logreg_metrics.json"
    transformer_path = REPORT_DIR / "transformer_metrics.json"
    
    if not baseline_path.exists() or not transformer_path.exists():
        print(f"⚠️  Laporan belum tersedia.")
        return
    
    baseline = load_metrics(baseline_path)
    transformer = load_metrics(transformer_path)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Title
    title_text = "MODEL SENTIMEN ANALYZER - SUMMARY"
    ax.text(0.5, 0.95, title_text, ha='center', fontsize=16, fontweight='bold',
            transform=ax.transAxes)
    
    # Baseline Info
    baseline_info = f"""
BASELINE MODEL (TF-IDF + Logistic Regression)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Dataset Size: {baseline.get('dataset_size', 'N/A'):,}
Test Size: {baseline.get('test_size', 'N/A'):,}

Metrics:
  • Accuracy: {baseline.get('accuracy', 0):.4f}
  • Macro F1: {baseline.get('macro_f1', 0):.4f}
  • Weighted F1: {baseline.get('weighted_f1', 0):.4f}

Model File: module_ML/models/baseline/
    """
    
    ax.text(0.05, 0.75, baseline_info, ha='left', va='top', fontsize=10,
            family='monospace', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8))
    
    # Transformer Info
    transformer_metrics = transformer.get('metrics', {})
    transformer_info = f"""
TRANSFORMER MODEL (IndoBERT)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Dataset Size: {transformer.get('dataset_size', 'N/A'):,}
Train Size: {transformer.get('train_size', 'N/A'):,}
Test Size: {transformer.get('test_size', 'N/A'):,}
Sampling: {transformer.get('sampling_strategy', 'N/A')}

Metrics:
  • Accuracy: {transformer_metrics.get('eval_accuracy', 0):.4f}
  • Macro F1: {transformer_metrics.get('eval_macro_f1', 0):.4f}
  • Weighted F1: {transformer_metrics.get('eval_weighted_f1', 0):.4f}

Model File: module_ML/models/transformer/final_model/
    """
    
    ax.text(0.05, 0.38, transformer_info, ha='left', va='top', fontsize=10,
            family='monospace', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='#ffe8e8', alpha=0.8))
    
    # Footer
    footer_text = "For more details, see: module_ML/reports/"
    ax.text(0.5, 0.02, footer_text, ha='center', fontsize=9, style='italic',
            transform=ax.transAxes, color='gray')
    
    plt.tight_layout()
    fig.savefig(REPORT_DIR / "model_summary.png", dpi=300, bbox_inches='tight')
    print(f"✓ Model summary card tersimpan: {REPORT_DIR}/model_summary.png")
    plt.close()


def generate_all_visualizations():
    """Generate semua visualisasi metrics."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("📊 Generating visualisasi metrik...\n")
    
    try:
        plot_metrics_comparison()
        plot_confusion_matrices()
        plot_model_summary()
        print("\n✅ Semua visualisasi berhasil dibuat!")
    except Exception as e:
        print(f"❌ Error saat membuat visualisasi: {e}")


if __name__ == "__main__":
    generate_all_visualizations()
