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
    """Visualize perbandingan metrik dari 3 baseline ML dan 1 model transformer."""
    report_paths = {
        "TF-IDF + LogReg": REPORT_DIR / "baseline_logreg_metrics.json",
        "TF-IDF + SVM": REPORT_DIR / "baseline_svm_metrics.json",
        "TF-IDF + Multinomial NB": REPORT_DIR / "baseline_nb_metrics.json",
        "Transformer (IndoBERT)": REPORT_DIR / "transformer_metrics.json",
    }

    missing = [name for name, path in report_paths.items() if not path.exists()]
    if missing:
        print(f"⚠️  Laporan belum tersedia untuk: {', '.join(missing)}. Jalankan train_run.py terlebih dahulu.")
        return

    reports = {name: load_metrics(path) for name, path in report_paths.items()}
    metrics_names = ["accuracy", "macro_f1", "weighted_f1"]

    model_scores = []
    for name, report in reports.items():
        if name == "Transformer (IndoBERT)":
            model_scores.append([report["metrics"].get(f"eval_{m}", 0) for m in metrics_names])
        else:
            model_scores.append([report.get(m, 0) for m in metrics_names])

    x = np.arange(len(metrics_names))
    width = 0.18

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = ["#3498db", "#2ecc71", "#f1c40f", "#e74c3c"]
    bars = []
    for i, (name, scores) in enumerate(zip(report_paths.keys(), model_scores)):
        bars.append(
            ax.bar(x + (i - 1.5) * width, scores, width, label=name, color=colors[i], alpha=0.85)
        )

    ax.set_xlabel("Metrik Evaluasi", fontsize=12, fontweight='bold')
    ax.set_ylabel("Skor", fontsize=12, fontweight='bold')
    ax.set_title("Perbandingan Performa: 3 Baseline ML vs Transformer", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics_names])
    ax.legend(fontsize=9)
    ax.set_ylim([0, 1.0])

    for bar_group in bars:
        for bar in bar_group:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    fig.savefig(REPORT_DIR / "metrics_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✓ Visualisasi metrics comparison tersimpan: {REPORT_DIR}/metrics_comparison.png")
    plt.close()


def plot_confusion_matrices():
    """Visualize confusion matrix untuk ketiga baseline dan transformer."""
    report_paths = {
        "TF-IDF + LogReg": REPORT_DIR / "baseline_logreg_metrics.json",
        "TF-IDF + SVM": REPORT_DIR / "baseline_svm_metrics.json",
        "TF-IDF + Multinomial NB": REPORT_DIR / "baseline_nb_metrics.json",
        "Transformer (IndoBERT)": REPORT_DIR / "transformer_metrics.json",
    }

    missing = [name for name, path in report_paths.items() if not path.exists()]
    if missing:
        print(f"⚠️  Laporan belum tersedia untuk: {', '.join(missing)}.")
        return

    reports = {name: load_metrics(path) for name, path in report_paths.items()}
    class_labels = ["Negatif", "Netral", "Positif"]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()

    for ax, (name, report) in zip(axes, reports.items()):
        if name == "Transformer (IndoBERT)":
            cm_data = report.get("metrics", {}).get("confusion_matrix")
        else:
            cm_data = report.get("confusion_matrix")

        if cm_data is None:
            ax.text(0.5, 0.5, "Tidak ada data confusion matrix", ha='center', va='center', fontsize=12)
            ax.axis('off')
            continue

        cm = np.array(cm_data)
        cmap = "Blues" if "TF-IDF" in name else "Reds"
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                    xticklabels=class_labels, yticklabels=class_labels, cbar=True)
        ax.set_title(f"Confusion Matrix: {name}", fontsize=12, fontweight='bold')
        ax.set_ylabel("Actual", fontsize=11)
        ax.set_xlabel("Predicted", fontsize=11)

    plt.tight_layout()
    fig.savefig(REPORT_DIR / "confusion_matrices.png", dpi=300, bbox_inches='tight')
    print(f"✓ Visualisasi confusion matrices tersimpan: {REPORT_DIR}/confusion_matrices.png")
    plt.close()


def plot_model_summary():
    """Create summary card dengan info semua model yang diuji."""
    report_paths = {
        "TF-IDF + LogReg": REPORT_DIR / "baseline_logreg_metrics.json",
        "TF-IDF + SVM": REPORT_DIR / "baseline_svm_metrics.json",
        "TF-IDF + Multinomial NB": REPORT_DIR / "baseline_nb_metrics.json",
        "Transformer (IndoBERT)": REPORT_DIR / "transformer_metrics.json",
    }

    missing = [name for name, path in report_paths.items() if not path.exists()]
    if missing:
        print(f"⚠️  Laporan belum tersedia untuk: {', '.join(missing)}.")
        return

    reports = {name: load_metrics(path) for name, path in report_paths.items()}
    baseline_report = reports["TF-IDF + LogReg"]
    dataset_size = baseline_report.get('dataset_size', 'N/A')
    test_size = baseline_report.get('test_size', 'N/A')

    def get_score(report, key):
        if report is reports["Transformer (IndoBERT)"]:
            return report.get("metrics", {}).get(f"eval_{key}", 0)
        return report.get(key, 0)

    rows = []
    for name, report in reports.items():
        rows.append(
            f"{name:23} | {get_score(report, 'accuracy'):6.4f} | {get_score(report, 'macro_f1'):6.4f} | {get_score(report, 'weighted_f1'):6.4f}"
        )

    summary_text = f"""
MODEL SUMMARY - 3 Baseline ML + 1 Transformer DL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Dataset Size: {dataset_size:,}
Test Size: {test_size:,}

Model                          | Accuracy | Macro F1 | Weighted F1
-------------------------------|----------|----------|-------------
{chr(10).join(rows)}

Notes:
  • SVM memberikan akurasi tertinggi di antara baseline ML.
  • Naive Bayes paling cepat untuk inferensi baseline.
  • IndoBERT transformer unggul dalam memahami konteks bahasa Indonesia.

Model files:
  • Baseline models: module_ML/models/baseline/
  • Transformer model: module_ML/models/transformer/final_model/
    """

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    ax.text(0.05, 0.95, "MODEL SENTIMEN ANALYZER - SUMMARY", ha='left', fontsize=16, fontweight='bold')
    ax.text(0.05, 0.90, summary_text, ha='left', va='top', fontsize=10, family='monospace')

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

        created = [
            (REPORT_DIR / "metrics_comparison.png").exists(),
            (REPORT_DIR / "confusion_matrices.png").exists(),
            (REPORT_DIR / "model_summary.png").exists(),
        ]
        if not all(created):
            raise RuntimeError("Beberapa visualisasi belum dapat dibuat karena laporan belum lengkap.")

        print("\n✅ Semua visualisasi berhasil dibuat!")
    except Exception as e:
        print(f"❌ Error saat membuat visualisasi: {e}")


if __name__ == "__main__":
    generate_all_visualizations()
