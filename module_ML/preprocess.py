"""Fungsi utilitas untuk memuat dan membersihkan data ulasan Tokopedia."""

import re
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

TEXT_CANDIDATE_COLUMNS = [
    "review",
    "review_text",
    "ulasan",
    "content",
    "text",
    "komentar",
]

LABEL_CANDIDATE_COLUMNS = [
    "sentiment",
    "sentiment_label",
    "label",
    "class",
    "kategori_sentimen",
    "sentimen",
]

LABEL_MAP: Dict[str, str] = {
    "positive": "positif",
    "negative": "negatif",
    "neutral": "netral",
    "positif": "positif",
    "negatif": "negatif",
    "netral": "netral",
    "pos": "positif",
    "neg": "negatif",
    "neu": "netral",
    "1": "positif",
    "0": "netral",
    "-1": "negatif",
}

VALID_LABELS = {"positif", "netral", "negatif"}


def simple_text_cleaning(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"https?://\\S+|www\\.\\S+", " ", text)
    text = re.sub(r"[^a-z0-9a-zA-Z\\s]", " ", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text


def infer_column(df: pd.DataFrame, candidates: list[str], kind: str) -> str:
    lower_map = {col.lower(): col for col in df.columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    raise ValueError(
        f"Kolom {kind} tidak ditemukan. Kandidat: {candidates}. Kolom tersedia: {list(df.columns)}"
    )


def normalize_label(raw_label: str) -> str:
    val = str(raw_label).lower().strip()
    normalized = LABEL_MAP.get(val, val)
    if normalized not in VALID_LABELS:
        raise ValueError(
            f"Label '{raw_label}' tidak dikenali. Pastikan label hanya positif/netral/negatif."
        )
    return normalized


def load_and_prepare_dataset(
    csv_path: str | Path,
    text_col: str | None = None,
    label_col: str | None = None,
) -> Tuple[pd.DataFrame, str, str]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"File dataset tidak ditemukan: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Dataset kosong.")

    text_col = text_col or infer_column(df, TEXT_CANDIDATE_COLUMNS, "teks")
    label_col = label_col or infer_column(df, LABEL_CANDIDATE_COLUMNS, "label")

    result = df[[text_col, label_col]].copy()
    result = result.dropna()
    result[text_col] = result[text_col].astype(str).map(simple_text_cleaning)
    result[label_col] = result[label_col].map(normalize_label)

    result = result[result[text_col].str.len() > 0].reset_index(drop=True)
    return result, text_col, label_col


def save_processed_dataset(df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
