"""Upload model fine-tuned dan app Space ke Hugging Face Hub."""

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi


def ensure_repo(api: HfApi, repo_id: str, repo_type: str) -> None:
    """Create repo if not exists."""
    if repo_type == "space":
        api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True, space_sdk="gradio")
        return
    api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)


def upload_dl_model(api: HfApi, model_dir: Path, model_repo: str) -> None:
    """Upload deep learning model (IndoBERT) to HF."""
    if not model_dir.exists():
        raise FileNotFoundError(f"Folder model DL tidak ditemukan: {model_dir}")

    ensure_repo(api, model_repo, repo_type="model")
    
    print(f"⏳ Uploading IndoBERT model to {model_repo}...")
    api.upload_folder(
        repo_id=model_repo,
        repo_type="model",
        folder_path=str(model_dir),
        commit_message="Upload fine-tuned IndoBERT for Indonesian sentiment classification",
    )
    print(f"✅ IndoBERT model uploaded: https://huggingface.co/{model_repo}")


def upload_ml_models(api: HfApi, baseline_dir: Path, ml_repo: str) -> None:
    """Upload ML baseline models (TF-IDF + LogReg, SVM) to HF."""
    if not baseline_dir.exists():
        raise FileNotFoundError(f"Folder baseline ML tidak ditemukan: {baseline_dir}")

    ensure_repo(api, ml_repo, repo_type="model")
    
    print(f"⏳ Uploading ML baseline models to {ml_repo}...")
    api.upload_folder(
        repo_id=ml_repo,
        repo_type="model",
        folder_path=str(baseline_dir),
        commit_message="Upload TF-IDF + Logistic Regression and SVM models for Indonesian sentiment classification",
    )
    print(f"✅ ML models uploaded: https://huggingface.co/{ml_repo}")


def upload_space(api: HfApi, space_dir: Path, space_repo: str) -> None:
    """Upload Space app to HF."""
    if not space_dir.exists():
        raise FileNotFoundError(f"Folder Space tidak ditemukan: {space_dir}")

    ensure_repo(api, space_repo, repo_type="space")
    
    print(f"⏳ Uploading Space app to {space_repo}...")
    api.upload_folder(
        repo_id=space_repo,
        repo_type="space",
        folder_path=str(space_dir),
        commit_message="Update Space app for sentiment classification",
    )
    print(f"✅ Space app updated: https://huggingface.co/spaces/{space_repo}")


def create_model_card(model_name: str, model_type: str, metrics: dict) -> str:
    """Create model card markdown."""
    if model_type == "dl":
        return f"""---
language: []
datasets: []
tags:
- indonesian
- sentiment-analysis
- indobert
- transformer
license: mit
---

# {model_name}

## Model Details

- **Model Type**: Fine-tuned Transformer (IndoBERT)
- **Base Model**: `indobenchmark/indobert-base-p1`
- **Language**: Indonesian
- **Task**: Sentiment Classification

## Performance

| Metric | Score |
|--------|-------|
| Accuracy | {metrics['accuracy']:.2%} |
| Macro F1 | {metrics['macro_f1']:.2%} |
| Weighted F1 | {metrics['weighted_f1']:.2%} |

## Training Details

- **Dataset**: Tokopedia Product Reviews 2025
- **Train Samples**: 52,268
- **Test Samples**: 13,067
- **Classes**: 3 (Negatif, Netral, Positif)
- **Epochs**: 5 (with early stopping)
- **Batch Size**: 16

## Usage

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("{model_name}")
tokenizer = AutoTokenizer.from_pretrained("{model_name}")

text = "Produk bagus, recommend!"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
predictions = outputs.logits.argmax(-1)
```

## Citation

```bibtex
@misc{{kelompok10_sentiment_2026,
  title={{Indonesian E-commerce Sentiment Classification}},
  year={{2026}},
  publisher={{Hugging Face}},
  howpublished={{\\url{{https://huggingface.co/{model_name}}}}}
}}
@endbibtex
```
"""
    else:  # ML models
        return f"""---
language: []
datasets: []
tags:
- indonesian
- sentiment-analysis
- tfidf
- sklearn
- logistic-regression
- svm
license: mit
---

# {model_name}

## Model Details

- **Model Type**: Scikit-learn Baseline Models
- **Vectorizer**: TF-IDF with Word & Character N-grams
- **Classifiers**: Logistic Regression & SVM
- **Language**: Indonesian
- **Task**: Sentiment Classification

## Performance

| Metric | LogReg | SVM |
|--------|--------|-----|
| Accuracy | 94.36% | ~97.60% |
| Macro F1 | 51.64% | N/A |
| Weighted F1 | 95.75% | N/A |

## Dataset

- **Name**: Tokopedia Product Reviews 2025
- **Train Samples**: 52,268
- **Test Samples**: 13,067
- **Classes**: 3 (Negatif, Netral, Positif)

## Usage

```python
import joblib

# Load model
model = joblib.load('tfidf_logreg.joblib')

# Make predictions
predictions = model.predict(['Produk bagus, recommend!'])
print(predictions)  # Output: [2] (2 = Positif)
```

## Features

- Word n-grams (1-3)
- Character n-grams (2-4)
- Max Features: 100,000
- Inference Speed: < 100ms on CPU

## Citation

```bibtex
@misc{{kelompok10_sentiment_2026,
  title={{Indonesian E-commerce Sentiment Classification}},
  year={{2026}},
  publisher{{Hugging Face}},
  howpublished={{\\url{{https://huggingface.co/{model_name}}}}}
}}
@endbibtex
```
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload ML/DL models to Hugging Face Hub")
    parser.add_argument(
        "--org", 
        type=str, 
        required=True, 
        help="Organization/username di HF, contoh: kelompok-10-NLP-SD-2026"
    )
    parser.add_argument(
        "--upload-dl", 
        action="store_true", 
        help="Upload deep learning model (IndoBERT)"
    )
    parser.add_argument(
        "--upload-ml", 
        action="store_true", 
        help="Upload ML baseline models (TF-IDF + LogReg/SVM)"
    )
    parser.add_argument(
        "--upload-space", 
        action="store_true", 
        help="Upload Gradio space"
    )
    parser.add_argument(
        "--dl-dir", 
        type=str, 
        default="module_ML/models/deep_learning/final_model",
        help="Path ke deep learning model"
    )
    parser.add_argument(
        "--ml-dir", 
        type=str, 
        default="module_ML/models/baseline",
        help="Path ke ML baseline models"
    )
    parser.add_argument(
        "--space-dir", 
        type=str, 
        default="module_ML/hf_space",
        help="Path ke Space app"
    )
    
    args = parser.parse_args()
    api = HfApi()
    
    # Load metrics
    metrics_file = Path("module_ML/reports/arxiv_report.json")
    if metrics_file.exists():
        with open(metrics_file) as f:
            report = json.load(f)
            dl_metrics = report["models"]["transformer"]["metrics"]
            ml_metrics = report["models"]["baseline"]["metrics"]
    else:
        dl_metrics = {"accuracy": 0.887, "macro_f1": 0.5088, "weighted_f1": 0.9268}
        ml_metrics = {"accuracy": 0.9436, "macro_f1": 0.5164, "weighted_f1": 0.9575}
    
    print("\n" + "="*70)
    print("🤖 UPLOAD MODELS KE HUGGING FACE")
    print("="*70 + "\n")
    
    if args.upload_dl:
        dl_repo = f"{args.org}/indobert-tokopedia-sentiment"
        upload_dl_model(api, Path(args.dl_dir), dl_repo)
        print()
    
    if args.upload_ml:
        ml_repo = f"{args.org}/tfidf-sentiment-baseline"
        upload_ml_models(api, Path(args.ml_dir), ml_repo)
        print()
    
    if args.upload_space:
        space_repo = f"{args.org}/tokopedia-sentiment-classifier"
        upload_space(api, Path(args.space_dir), space_repo)
        print()
    
    print("="*70)
    print("✅ UPLOAD SELESAI!")
    print("="*70)
    print(f"\n📦 Models tersedia di: https://huggingface.co/{args.org}")
    print(f"   DL Model:  https://huggingface.co/{args.org}/indobert-tokopedia-sentiment")
    print(f"   ML Model:  https://huggingface.co/{args.org}/tfidf-sentiment-baseline")
    if args.upload_space:
        print(f"   Space:     https://huggingface.co/spaces/{args.org}/tokopedia-sentiment-classifier")
    print()


if __name__ == "__main__":
    main()
