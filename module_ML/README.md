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
