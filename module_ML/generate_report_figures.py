"""Generate gambar ringkas untuk README dari file report evaluasi."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from config import REPORT_DIR

FIGURE_DIR = REPORT_DIR / "figures"


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_metric(payload: dict, metric_name: str) -> float | None:
    if not payload:
        return None

    if metric_name in payload:
        value = payload.get(metric_name)
        return float(value) if value is not None else None

    metrics = payload.get("metrics", {})
    eval_key = f"eval_{metric_name}"
    if eval_key in metrics:
        value = metrics.get(eval_key)
        return float(value) if value is not None else None
    if metric_name in metrics:
        value = metrics.get(metric_name)
        return float(value) if value is not None else None
    return None


def save_metric_comparison() -> None:
    reports = {
        "LogReg": load_json(REPORT_DIR / "baseline_logreg_metrics.json"),
        "SVM": load_json(REPORT_DIR / "baseline_svm_metrics.json"),
        "IndoBERT": load_json(REPORT_DIR / "transformer_metrics.json"),
    }

    metrics = ["accuracy", "macro_f1", "weighted_f1"]
    labels = list(reports.keys())
    values = {
        metric: [extract_metric(report, metric) for report in reports.values()]
        for metric in metrics
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(labels))
    width = 0.24
    palette = sns.color_palette("deep", n_colors=len(metrics))

    for idx, metric in enumerate(metrics):
        offset = (idx - 1) * width
        ax.bar(
            [pos + offset for pos in x],
            values[metric],
            width=width,
            label=metric.replace("_", " ").title(),
            color=palette[idx],
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Skor")
    ax.set_title("Perbandingan Performa Model")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "metric_comparison.png", dpi=200)
    plt.close(fig)


def save_confusion_matrix(model_name: str, payload: dict) -> None:
    cm = payload.get("confusion_matrix")
    labels = payload.get("label_order", ["negatif", "netral", "positif"])
    if not cm:
        return

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Label Asli")
    ax.set_title(f"Confusion Matrix - {model_name}")
    fig.tight_layout()
    filename = f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
    fig.savefig(FIGURE_DIR / filename, dpi=200)
    plt.close(fig)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    save_metric_comparison()
    save_confusion_matrix("baseline_logreg", load_json(REPORT_DIR / "baseline_logreg_metrics.json"))
    save_confusion_matrix("baseline_svm", load_json(REPORT_DIR / "baseline_svm_metrics.json"))
    save_confusion_matrix("transformer", load_json(REPORT_DIR / "transformer_metrics.json"))
    print(f"Figure tersimpan di: {FIGURE_DIR}")


if __name__ == "__main__":
    main()