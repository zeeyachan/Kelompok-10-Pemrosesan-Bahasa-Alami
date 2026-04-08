"""Unduh dataset Kaggle Tokopedia Product Reviews 2025 ke folder data/raw."""

from pathlib import Path

import kagglehub

from config import RAW_DATA_DIR

KAGGLE_DATASET_ID = "salmanabdu/tokopedia-product-reviews-2025"


def main() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Mengunduh dataset: {KAGGLE_DATASET_ID}")
    download_path = Path(kagglehub.dataset_download(KAGGLE_DATASET_ID))
    print(f"Dataset tersimpan sementara di: {download_path}")

    copied_files = []
    for file_path in download_path.rglob("*"):
        if file_path.is_file():
            target = RAW_DATA_DIR / file_path.name
            target.write_bytes(file_path.read_bytes())
            copied_files.append(target)

    print("File berhasil disalin ke data/raw:")
    for f in copied_files:
        print(f"- {f}")


if __name__ == "__main__":
    main()
