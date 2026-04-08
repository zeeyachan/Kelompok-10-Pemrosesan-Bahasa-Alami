"""Inference CLI untuk baseline dan transformer."""

import argparse
from pathlib import Path

import joblib
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from preprocess import simple_text_cleaning

DEFAULT_ID2LABEL = {0: "negatif", 1: "netral", 2: "positif"}


def resolve_id2label(model) -> dict[int, str]:
    raw_id2label = getattr(model.config, "id2label", None) or {}
    resolved: dict[int, str] = {}
    for idx, default_label in DEFAULT_ID2LABEL.items():
        raw_label = raw_id2label.get(idx)
        if raw_label is None:
            raw_label = raw_id2label.get(str(idx))
        if isinstance(raw_label, str) and not raw_label.startswith("LABEL_"):
            resolved[idx] = raw_label
        else:
            resolved[idx] = default_label
    return resolved


def predict_baseline(text: str, model_path: str) -> str:
    model = joblib.load(model_path)
    pred = model.predict([simple_text_cleaning(text)])[0]
    return str(pred)


def predict_transformer(text: str, model_dir: str) -> tuple[str, float]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    id2label = resolve_id2label(model)

    encoded = tokenizer(
        simple_text_cleaning(text),
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )

    with torch.no_grad():
        logits = model(**encoded).logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred_id = int(torch.argmax(probs).item())

    return id2label.get(pred_id, str(pred_id)), float(probs[pred_id].item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "transformer"], required=True)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--model-path", type=str, default=str(Path("module_ML/models/baseline/tfidf_logreg.joblib")))
    parser.add_argument("--model-dir", type=str, default=str(Path("module_ML/models/transformer/final_model")))
    args = parser.parse_args()

    if args.mode == "baseline":
        label = predict_baseline(args.text, args.model_path)
        print({"label": label})
    else:
        label, score = predict_transformer(args.text, args.model_dir)
        print({"label": label, "confidence": score})


if __name__ == "__main__":
    main()
