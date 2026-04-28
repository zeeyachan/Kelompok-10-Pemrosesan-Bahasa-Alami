"""Fine-tuning IndoBERT untuk klasifikasi sentimen ulasan Tokopedia."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from config import (
    DEFAULT_DATASET_CSV,
    MODEL_DIR,
    RANDOM_STATE,
    REPORT_DIR,
    TRANSFORMER_BATCH_SIZE,
    TRANSFORMER_LR,
    TRANSFORMER_MAX_LENGTH,
    TRANSFORMER_MODEL_NAME,
    TRANSFORMER_NUM_EPOCHS,
    TRANSFORMER_WARMUP_RATIO,
    TRANSFORMER_EARLY_STOPPING_PATIENCE,
)
from preprocess import load_and_prepare_dataset

LABEL2ID = {"negatif": 0, "netral": 1, "positif": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fn(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    cm = confusion_matrix(labels, preds, labels=[0, 1, 2])
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "weighted_f1": f1_score(labels, preds, average="weighted"),
        "confusion_matrix": cm.tolist(),
    }


def sample_train_per_class(df, max_samples_per_class: int):
    chunks = []
    for label_id in sorted(LABEL2ID.values()):
        subset = df[df["labels"] == label_id]
        if subset.empty:
            raise ValueError(f"Kelas {ID2LABEL[label_id]} tidak ada di train split.")
        n = min(len(subset), max_samples_per_class)
        chunks.append(subset.sample(n=n, random_state=RANDOM_STATE, replace=False))

    sampled = pd.concat(chunks, ignore_index=True)
    return sampled.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=str(DEFAULT_DATASET_CSV))
    parser.add_argument("--text-col", type=str, default=None)
    parser.add_argument("--label-col", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=TRANSFORMER_MODEL_NAME)
    parser.add_argument("--output-dir", type=str, default=str(MODEL_DIR / "transformer"))
    parser.add_argument("--epochs", type=int, default=TRANSFORMER_NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=TRANSFORMER_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=TRANSFORMER_MAX_LENGTH)
    parser.add_argument("--learning-rate", type=float, default=TRANSFORMER_LR)
    parser.add_argument(
        "--eval-each-epoch",
        action="store_true",
        default=True,
        help="Lakukan evaluasi dan save setiap epoch (default: True).",
    )
    sample_group = parser.add_mutually_exclusive_group()
    sample_group.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Batasi jumlah data train untuk eksperimen cepat (stratified).",
    )
    sample_group.add_argument(
        "--max-samples-per-class",
        type=int,
        default=None,
        help="Batasi jumlah data train per kelas agar lebih seimbang.",
    )
    parser.add_argument(
        "--eval-max-samples",
        type=int,
        default=None,
        help="Batasi jumlah data evaluasi (stratified) untuk eksperimen cepat.",
    )
    args = parser.parse_args()

    df, text_col, label_col = load_and_prepare_dataset(args.csv, args.text_col, args.label_col)
    df = df.copy()
    df["labels"] = df[label_col].map(LABEL2ID)

    train_df, test_df = train_test_split(
        df[[text_col, "labels"]],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["labels"],
    )

    sampling_strategy = "full_train"
    if args.max_samples_per_class is not None and args.max_samples_per_class > 0:
        train_df = sample_train_per_class(train_df, args.max_samples_per_class)
        sampling_strategy = f"per_class_{args.max_samples_per_class}"
    elif args.max_samples is not None and args.max_samples > 0 and args.max_samples < len(train_df):
        train_df, _ = train_test_split(
            train_df,
            train_size=args.max_samples,
            random_state=RANDOM_STATE,
            stratify=train_df["labels"],
        )
        sampling_strategy = f"stratified_train_{args.max_samples}"

    eval_sampling_strategy = "full_test"
    if args.eval_max_samples is not None and args.eval_max_samples > 0 and args.eval_max_samples < len(test_df):
        test_df, _ = train_test_split(
            test_df,
            train_size=args.eval_max_samples,
            random_state=RANDOM_STATE,
            stratify=test_df["labels"],
        )
        eval_sampling_strategy = f"stratified_test_{args.eval_max_samples}"

    train_ds = Dataset.from_pandas(train_df.rename(columns={text_col: "text"}), preserve_index=False)
    test_ds = Dataset.from_pandas(test_df.rename(columns={text_col: "text"}), preserve_index=False)

    class_counts = train_df["labels"].value_counts().reindex(sorted(LABEL2ID.values()), fill_value=0)
    if (class_counts == 0).any():
        missing_labels = [ID2LABEL[label_id] for label_id, count in class_counts.items() if count == 0]
        raise ValueError(
            "Semua kelas harus ada di data train setelah split stratified. "
            f"Kelas yang hilang: {missing_labels}"
        )

    class_weights = class_counts.sum() / (len(class_counts) * class_counts.astype(float))
    class_weights_tensor = torch.tensor(class_weights.to_list(), dtype=torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    train_ds = train_ds.map(tokenize_fn, batched=True)
    test_ds = test_ds.map(tokenize_fn, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_steps=int(500 * args.epochs),
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=RANDOM_STATE,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        class_weights=class_weights_tensor,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=TRANSFORMER_EARLY_STOPPING_PATIENCE,
                early_stopping_threshold=0.0,
            )
        ],
    )

    trainer.train()
    eval_metrics = trainer.evaluate()

    final_dir = output_dir / "final_model"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "model_name": args.model_name,
        "dataset_size": len(df),
        "train_size": len(train_df),
        "test_size": len(test_df),
        "sampling_strategy": sampling_strategy,
        "eval_sampling_strategy": eval_sampling_strategy,
        "text_col": text_col,
        "label_col": label_col,
        "class_weights": {
            ID2LABEL[idx]: float(weight) for idx, weight in zip(class_counts.index, class_weights_tensor.tolist())
        },
        "metrics": eval_metrics,
    }

    report_path = REPORT_DIR / "transformer_metrics.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False)

    print(f"Model transformer tersimpan di: {final_dir}")
    print(f"Laporan evaluasi tersimpan di: {report_path}")


if __name__ == "__main__":
    main()
