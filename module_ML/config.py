"""Konfigurasi global untuk proyek analisis sentimen Tokopedia."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODULE_ROOT = PROJECT_ROOT / "module_ML"
DATA_DIR = MODULE_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = MODULE_ROOT / "models"
REPORT_DIR = MODULE_ROOT / "reports"

# Ubah sesuai nama file hasil download dataset Kaggle.
DEFAULT_DATASET_CSV = RAW_DATA_DIR / "tokopedia_product_reviews_2025.csv"

RANDOM_STATE = 42
BASELINE_TEST_SIZE = 0.2
BASELINE_MAX_FEATURES = 50000

TRANSFORMER_MODEL_NAME = "indobenchmark/indobert-base-p1"
TRANSFORMER_MAX_LENGTH = 256
TRANSFORMER_BATCH_SIZE = 16
TRANSFORMER_NUM_EPOCHS = 3
TRANSFORMER_LR = 2e-5
TRANSFORMER_WARMUP_RATIO = 0.1
