"""Menjalankan eksperimen baseline + transformer sekaligus dan membuat ringkasan perbandingan."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from config import DEFAULT_DATASET_CSV, MODULE_ROOT, REPORT_DIR


def run_command(cmd: list[str]) -> None:
    print("Menjalankan:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pick_metric(payload: dict) -> float | None:
    if not payload:
        return None

    if "macro_f1" in payload:
        return payload.get("macro_f1")

    metrics = payload.get("metrics", {})
    if "eval_macro_f1" in metrics:
        return metrics.get("eval_macro_f1")
    if "macro_f1" in metrics:
        return metrics.get("macro_f1")
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=str(DEFAULT_DATASET_CSV))
    parser.add_argument("--text-col", type=str, default=None)
    parser.add_argument("--label-col", type=str, default=None)
    parser.add_argument("--run-baseline", action="store_true")
    parser.add_argument("--run-transformer", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-samples-per-class", type=int, default=None)
    parser.add_argument("--eval-max-samples", type=int, default=None)
    args = parser.parse_args()
    python_exec = sys.executable

    run_baseline = args.run_baseline or (not args.run_baseline and not args.run_transformer)
    run_transformer = args.run_transformer or (not args.run_baseline and not args.run_transformer)

    if run_baseline:
        for algo in ["logreg", "svm", "nb"]:
            cmd = [
                python_exec,
                str(MODULE_ROOT / "train_baseline.py"),
                "--csv",
                args.csv,
                "--algo",
                algo,
            ]
            if args.text_col:
                cmd.extend(["--text-col", args.text_col])
            if args.label_col:
                cmd.extend(["--label-col", args.label_col])
            run_command(cmd)

    if run_transformer:
        cmd = [
            python_exec,
            str(MODULE_ROOT / "train_transformer.py"),
            "--csv",
            args.csv,
        ]
        if args.text_col:
            cmd.extend(["--text-col", args.text_col])
        if args.label_col:
            cmd.extend(["--label-col", args.label_col])
        if args.epochs is not None:
            cmd.extend(["--epochs", str(args.epochs)])
        if args.batch_size is not None:
            cmd.extend(["--batch-size", str(args.batch_size)])
        if args.max_length is not None:
            cmd.extend(["--max-length", str(args.max_length)])
        if args.learning_rate is not None:
            cmd.extend(["--learning-rate", str(args.learning_rate)])
        if args.max_samples is not None:
            cmd.extend(["--max-samples", str(args.max_samples)])
        if args.max_samples_per_class is not None:
            cmd.extend(["--max-samples-per-class", str(args.max_samples_per_class)])
        if args.eval_max_samples is not None:
            cmd.extend(["--eval-max-samples", str(args.eval_max_samples)])
        run_command(cmd)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    baseline_logreg = load_json(REPORT_DIR / "baseline_logreg_metrics.json")
    baseline_svm = load_json(REPORT_DIR / "baseline_svm_metrics.json")
    transformer = load_json(REPORT_DIR / "transformer_metrics.json")

    summary = {
        "baseline_logreg_macro_f1": pick_metric(baseline_logreg),
        "baseline_svm_macro_f1": pick_metric(baseline_svm),
        "baseline_nb_macro_f1": pick_metric(load_json(REPORT_DIR / "baseline_nb_metrics.json")),
        "transformer_indobert_macro_f1": pick_metric(transformer),
    }

    summary_path = REPORT_DIR / "experiment_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Ringkasan perbandingan metrik disimpan di:", summary_path)
    print(summary)
    
    # Generate visualisasi
    print("\n" + "="*60)
    print("📊 Generating visualisasi metrics...")
    print("="*60)
    try:
        from visualize_metrics import generate_all_visualizations
        generate_all_visualizations()
    except Exception as e:
        print(f"⚠️  Visualisasi tidak dapat dibuat: {e}")


if __name__ == "__main__":
    main()
