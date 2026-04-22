---
title: Sentiment Analysis Tokopedia
colorFrom: yellow
colorTo: blue
sdk: gradio
sdk_version: 5.24.0
app_file: app.py
pinned: false
---

# Sentiment Analysis Tokopedia (IndoBERT)

Aplikasi Gradio untuk klasifikasi sentimen ulasan produk Tokopedia menjadi:

- positif
- netral
- negatif

## Cara pakai di Hugging Face Spaces

1. Upload file `app.py` dan `requirements.txt` ke Space.
2. Set Environment Variable `MODEL_REPO` ke repo model hasil fine-tuning Anda, misalnya:
   `username/indobert-tokopedia-sentiment`
3. Deploy Space.

Jika `MODEL_REPO` belum diatur atau gagal dimuat, aplikasi akan mencoba fallback ke base IndoBERT agar Space tetap dapat berjalan untuk smoke test.

## Upload Otomatis (Opsional)

Jika sudah login Hugging Face (`huggingface-cli login`), Anda juga bisa upload model + update Space langsung dari root project:

```bash
python module_ML/deploy_hf.py \
   --model-repo username/indobert-tokopedia-sentiment \
   --space-repo username/tokopedia-sentiment-space
```
