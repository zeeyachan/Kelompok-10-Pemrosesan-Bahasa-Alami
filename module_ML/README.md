# Module ML - Analisis Sentimen Tokopedia

Modul ini berisi pipeline end-to-end untuk:

- baseline TF-IDF + Logistic Regression/SVM
- fine-tuning IndoBERT
- perbandingan metrik performa
- deployment Hugging Face Spaces

## Jalankan cepat

```bash
pip install -r module_ML/requirements.txt
python module_ML/download_data.py
python module_ML/train_run.py --csv module_ML/data/raw/tokopedia_product_reviews_2025.csv
```

## Output penting

- Model baseline: `module_ML/models/baseline/`
- Model IndoBERT: `module_ML/models/transformer/final_model/`
- Report evaluasi: `module_ML/reports/`

## Catatan Model

- Baseline TF-IDF + Logistic Regression/SVM sudah memakai `class_weight="balanced"`.
- Fine-tuning IndoBERT juga memakai weighted loss agar kelas minoritas tetap terakomodasi.

## Deploy ke Hugging Face Hub

Login terlebih dahulu:

```bash
huggingface-cli login
```

Upload model hasil fine-tuning:

```bash
python module_ML/deploy_hf.py --model-repo username/indobert-tokopedia-sentiment
```

Upload model + update Space sekaligus:

```bash
python module_ML/deploy_hf.py \
	--model-repo username/indobert-tokopedia-sentiment \
	--space-repo username/tokopedia-sentiment-space
```

## Rekomendasi Training IndoBERT (Imbalance)

Karena distribusi label sangat timpang, jalankan training dengan sampling per-kelas agar model tidak bias ke kelas mayoritas:

```bash
python module_ML/train_transformer.py \
	--csv module_ML/data/raw/tokopedia_product_reviews_2025.csv \
	--epochs 3 \
	--batch-size 16 \
	--max-length 128 \
	--max-samples-per-class 600
```

Untuk eksperimen cepat, evaluasi bisa dibatasi agar runtime lebih singkat:

```bash
python module_ML/train_transformer.py \
	--csv module_ML/data/raw/tokopedia_product_reviews_2025.csv \
	--epochs 1 \
	--batch-size 16 \
	--max-length 128 \
	--max-samples-per-class 120 \
	--eval-max-samples 600
```
