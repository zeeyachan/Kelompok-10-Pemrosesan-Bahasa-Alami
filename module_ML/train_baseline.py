"""Training baseline sentiment classifier berbasis TF-IDF + Logistic Regression/SVM."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC

from config import (
    BASELINE_MAX_FEATURES,
    BASELINE_TEST_SIZE,
    DEFAULT_DATASET_CSV,
    MODEL_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    REPORT_DIR,
)
from preprocess import load_and_prepare_dataset, save_processed_dataset


def build_model(algo: str) -> Pipeline:
    if algo == "logreg":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE, C=0.5)
    elif algo == "svm":
        clf = LinearSVC(class_weight="balanced", random_state=RANDOM_STATE, C=0.5)
    else:
        raise ValueError("Algoritma tidak valid. Gunakan: logreg atau svm")

    # Gunakan FeatureUnion untuk menggabungkan word n-grams dan character n-grams
    feature_union = FeatureUnion([
        ("word_tfidf", TfidfVectorizer(
            ngram_range=(1, 3),
            min_df=2,
            max_features=BASELINE_MAX_FEATURES // 2,
            sublinear_tf=True,
            strip_accents='unicode',
            lowercase=True,
        )),
        ("char_tfidf", TfidfVectorizer(
            ngram_range=(2, 4),
            min_df=2,
            max_features=BASELINE_MAX_FEATURES // 2,
            sublinear_tf=True,
            analyzer='char',
            strip_accents='unicode',
            lowercase=True,
        )),
    ])

    return Pipeline([
        ("features", feature_union),
        ("classifier", clf),
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=str(DEFAULT_DATASET_CSV))
    parser.add_argument("--algo", type=str, default="logreg", choices=["logreg", "svm"])
    parser.add_argument("--text-col", type=str, default=None)
    parser.add_argument("--label-col", type=str, default=None)
    args = parser.parse_args()

    df, text_col, label_col = load_and_prepare_dataset(args.csv, args.text_col, args.label_col)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    processed_path = PROCESSED_DATA_DIR / "tokopedia_reviews_processed.csv"
    save_processed_dataset(df, processed_path)

    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col],
        df[label_col],
        test_size=BASELINE_TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[label_col],
    )

    model = build_model(args.algo)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "algorithm": args.algo,
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
        "weighted_f1": f1_score(y_test, y_pred, average="weighted"),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "test_size": len(X_test),
        "dataset_size": len(df),
        "text_col": text_col,
        "label_col": label_col,
    }

    model_out_dir = MODEL_DIR / "baseline"
    model_out_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_out_dir / f"tfidf_{args.algo}.joblib"
    joblib.dump(model, model_path)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / f"baseline_{args.algo}_metrics.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"Model baseline tersimpan di: {model_path}")
    print(f"Laporan evaluasi tersimpan di: {report_path}")
    print("Ringkasan metrik:")
    print(pd.Series({k: v for k, v in metrics.items() if isinstance(v, (int, float, str))}))


if __name__ == "__main__":
    main()
