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

Jika `MODEL_REPO` belum diatur atau gagal dimuat, aplikasi akan mencoba fallback ke model sentimen publik agar Space tetap dapat berjalan.
