"""Generate visualization untuk metrik evaluasi model baseline dan transformer."""

import json
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

from config import REPORT_DIR

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 5)

ARXIV_REPORT_PATH = REPORT_DIR / "arxiv_report.json"


def load_json(report_path: Path) -> Dict[str, Any]:
    if not report_path.exists():
        return {}
    with report_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_metrics(report_path: Path, model_key: str | None = None) -> Dict[str, Any]:
    """Load metrics dari file JSON atau fallback ke arxiv_report.json."""
    data = load_json(report_path)
    if data:
        return data

    if model_key is None:
        return {}

    arxiv_report = load_json(ARXIV_REPORT_PATH)
    return arxiv_report.get("models", {}).get(model_key, {})


def plot_metrics_comparison():
    """Visualize perbandingan metrik dari 3 baseline ML dan 1 model transformer."""
    report_mapping = {
        "TF-IDF + LogReg": (REPORT_DIR / "baseline_logreg_metrics.json", "baseline_logreg"),
        "TF-IDF + SVM": (REPORT_DIR / "baseline_svm_metrics.json", "baseline_svm"),
        "TF-IDF + Multinomial NB": (REPORT_DIR / "baseline_nb_metrics.json", "baseline_nb"),
        "Transformer (IndoBERT)": (REPORT_DIR / "transformer_metrics.json", "transformer"),
    }

    if not ARXIV_REPORT_PATH.exists() and any(not path.exists() for path, _ in report_mapping.values()):
        missing = [name for name, (path, _) in report_mapping.items() if not path.exists()]
        print(f"⚠️  Laporan belum tersedia untuk: {', '.join(missing)}. Jalankan train_run.py terlebih dahulu.")
        return

    reports = {
        name: load_metrics(path, model_key=model_key)
        for name, (path, model_key) in report_mapping.items()
    }

    metrics_names = ["accuracy", "macro_f1", "weighted_f1"]

    def score(report: Dict[str, Any], key: str) -> float:
        if "metrics" in report:
            return report["metrics"].get(key, 0)
        return report.get(key, 0)

    model_scores = [[score(report, m) for m in metrics_names] for report in reports.values()]

    x = np.arange(len(metrics_names))
    width = 0.18

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = ["#3498db", "#2ecc71", "#f1c40f", "#e74c3c"]
    bars = []
    for i, (name, scores) in enumerate(zip(reports.keys(), model_scores)):
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
    return True


def plot_confusion_matrices():
    """Visualize confusion matrix untuk ketiga baseline dan transformer."""
    report_paths = {
        "TF-IDF + LogReg": (REPORT_DIR / "baseline_logreg_metrics.json", "baseline_logreg"),
        "TF-IDF + SVM": (REPORT_DIR / "baseline_svm_metrics.json", "baseline_svm"),
        "TF-IDF + Multinomial NB": (REPORT_DIR / "baseline_nb_metrics.json", "baseline_nb"),
        "Transformer (IndoBERT)": (REPORT_DIR / "transformer_metrics.json", "transformer"),
    }

    has_any_confusion = False
    reports = {}
    for name, (path, model_key) in report_paths.items():
        report = load_metrics(path, model_key=model_key)
        reports[name] = report
        if name != "Transformer (IndoBERT)" and report.get("confusion_matrix") is not None:
            has_any_confusion = True
        if name == "Transformer (IndoBERT)" and report.get("metrics", {}).get("confusion_matrix") is not None:
            has_any_confusion = True

    if not has_any_confusion:
        print("⚠️  Data confusion matrix tidak tersedia di laporan. Confusion matrices tidak dapat dibuat.")
        return False

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
    return True


def plot_model_summary() -> bool:
    """Create summary card dengan info semua model yang diuji."""
    report_mapping = {
        "TF-IDF + LogReg": (REPORT_DIR / "baseline_logreg_metrics.json", "baseline_logreg"),
        "TF-IDF + SVM": (REPORT_DIR / "baseline_svm_metrics.json", "baseline_svm"),
        "TF-IDF + Multinomial NB": (REPORT_DIR / "baseline_nb_metrics.json", "baseline_nb"),
        "Transformer (IndoBERT)": (REPORT_DIR / "transformer_metrics.json", "transformer"),
    }

    arxiv_report = load_json(ARXIV_REPORT_PATH)
    reports = {
        name: load_metrics(path, model_key=model_key)
        for name, (path, model_key) in report_mapping.items()
    }

    if not any(reports.values()) and not arxiv_report:
        print("⚠️  Laporan model tidak tersedia, tidak dapat membuat model summary.")
        return False

    dataset_size = arxiv_report.get('dataset', {}).get('total_samples', 'N/A')
    test_size = arxiv_report.get('dataset', {}).get('test_samples', 'N/A')

    def get_score(report: Dict[str, Any], key: str) -> float:
        if "metrics" in report:
            return report["metrics"].get(key, 0)
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
    return True


def generate_all_visualizations():
    """Generate semua visualisasi metrics."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("📊 Generating visualisasi metrik...\n")
    try:
        metrics_ok = plot_metrics_comparison()
        confusion_ok = plot_confusion_matrices()
        summary_ok = plot_model_summary()

        if metrics_ok and summary_ok:
            print("\n✅ Metrics comparison dan model summary berhasil dibuat!")
        else:
            print("\n⚠️ Beberapa visualisasi belum berhasil dibuat.")

        if not confusion_ok:
            print("⚠️ Data confusion matrix tidak tersedia, sehingga confusion_matrices.png tidak dibuat atau diperbarui.")
    except Exception as e:
        print(f"❌ Error saat membuat visualisasi: {e}")


if __name__ == "__main__":
    generate_all_visualizations()
