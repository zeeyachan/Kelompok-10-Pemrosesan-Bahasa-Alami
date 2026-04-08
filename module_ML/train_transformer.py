"""Fine-tuning IndoBERT untuk klasifikasi sentimen ulasan Tokopedia."""

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
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
)
from preprocess import load_and_prepare_dataset

LABEL2ID = {"negatif": 0, "netral": 1, "positif": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "weighted_f1": f1_score(labels, preds, average="weighted"),
    }


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
        "--max-samples",
        type=int,
        default=None,
        help="Batasi jumlah data untuk eksperimen cepat (stratified).",
    )
    args = parser.parse_args()

    df, text_col, label_col = load_and_prepare_dataset(args.csv, args.text_col, args.label_col)
    df = df.copy()
    df["labels"] = df[label_col].map(LABEL2ID)

    if args.max_samples is not None and args.max_samples > 0 and args.max_samples < len(df):
        df, _ = train_test_split(
            df,
            train_size=args.max_samples,
            random_state=RANDOM_STATE,
            stratify=df["labels"],
        )

    train_df, test_df = train_test_split(
        df[[text_col, "labels"]],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["labels"],
    )

    train_ds = Dataset.from_pandas(train_df.rename(columns={text_col: "text"}), preserve_index=False)
    test_ds = Dataset.from_pandas(test_df.rename(columns={text_col: "text"}), preserve_index=False)

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
        warmup_ratio=TRANSFORMER_WARMUP_RATIO,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=RANDOM_STATE,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
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
        "test_size": len(test_df),
        "text_col": text_col,
        "label_col": label_col,
        "metrics": eval_metrics,
    }

    report_path = REPORT_DIR / "transformer_metrics.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False)

    print(f"Model transformer tersimpan di: {final_dir}")
    print(f"Laporan evaluasi tersimpan di: {report_path}")


if __name__ == "__main__":
    main()
