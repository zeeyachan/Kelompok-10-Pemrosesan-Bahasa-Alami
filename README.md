# Kelompok-10-Pemrosesan-Bahasa-Alami

Tugas besar mata kuliah Pemrosesan Bahasa Alami Sains Data ITERA 2026

## Anggota Tim

- <img src="https://github.com/zeeyachan.png" width="22"> **Nabila Zakiyah Zahra** (122450139) — [@zeeyachan](https://github.com/zeeyachan)

- <img src="https://github.com/salwaf01.png" width="22"> **Salwa Farhanatussaidah** (122450055) — [@salwaf01](https://github.com/salwaf01)

- <img src="https://github.com/nasywanaff.png" width="22"> **Nasywa Nur Afifah** (122450125) — [@nasywanaff](https://github.com/nasywanaff)


## Dataset
 
Dataset yang digunakan berasal dari Kaggle:  
https://www.kaggle.com/datasets/salmanabdu/tokopedia-product-reviews-2025  

Dataset ini berisi **ulasan produk Tokopedia** yang digunakan untuk analisis sentimen dalam Bahasa Indonesia.

### Deskripsi
- Jenis data: teks ulasan pengguna  
- Bahasa: Indonesia  
- Domain: e-commerce  
- Tujuan: klasifikasi sentimen (positif, netral, negatif)  

### Struktur Data
Kolom utama pada dataset:
- `review_text` : isi ulasan pengguna  
- `rating` : nilai rating (1–5)  
- `label` : kategori sentimen  

Contoh data:

| review_text | rating | label |
|------------|--------|-------|
| Barang bagus, sesuai deskripsi | 5 | positif |
| Pengiriman lama, barang kurang rapi | 2 | negatif |

### Catatan
- Data berupa teks tidak formal (mengandung typo, singkatan, dll)  
- Dilakukan preprocessing sebelum digunakan dalam model  

## Tujuan Proyek

Proyek ini membangun sistem analisis sentimen ulasan produk Tokopedia untuk klasifikasi:

- positif
- netral
- negatif

Pendekatan yang digunakan:

- Baseline machine learning: TF-IDF + Logistic Regression / SVM
- Model utama transformer: IndoBERT (`indobenchmark/indobert-base-p1`)

Hasil kedua pendekatan dibandingkan menggunakan metrik evaluasi (accuracy, macro-F1, weighted-F1).

## Struktur Proyek

```text
module_ML/
	config.py
	preprocess.py
	download_data.py
	train_baseline.py
	train_transformer.py
	train_run.py
	predict.py
	requirements.txt
	hf_space/
		app.py
		requirements.txt
		README.md
```

## Setup Environment

1. Buat virtual environment.
2. Install dependency:

```bash
pip install -r module_ML/requirements.txt
```

## Download Dataset Kaggle

Pastikan akun Kaggle sudah terkonfigurasi (token API aktif), lalu jalankan:

```bash
python module_ML/download_data.py
```

Secara default file akan disalin ke folder `module_ML/data/raw/`.

## Menjalankan Training

### 1) Training Baseline (TF-IDF + Logistic Regression)

```bash
python module_ML/train_baseline.py --algo logreg --csv module_ML/data/raw/tokopedia_product_reviews_2025.csv
```

### 2) Training Baseline (TF-IDF + SVM)

```bash
python module_ML/train_baseline.py --algo svm --csv module_ML/data/raw/tokopedia_product_reviews_2025.csv
```

### 3) Fine-tuning IndoBERT

```bash
python module_ML/train_transformer.py --csv module_ML/data/raw/tokopedia_product_reviews_2025.csv
```

### 4) Menjalankan Semua Eksperimen Sekaligus

```bash
python module_ML/train_run.py --csv module_ML/data/raw/tokopedia_product_reviews_2025.csv
```

Output evaluasi akan tersimpan di folder `module_ML/reports/`.

## Inferensi Cepat

### Baseline

```bash
python module_ML/predict.py --mode baseline --model-path module_ML/models/baseline/tfidf_logreg.joblib --text "Barang bagus, pengiriman cepat"
```

### IndoBERT

```bash
python module_ML/predict.py --mode transformer --model-dir module_ML/models/transformer/final_model --text "Barang sesuai deskripsi dan seller ramah"
```

## Deploy ke Hugging Face Spaces

Folder siap deploy tersedia di `module_ML/hf_space/`.

Langkah deployment:

1. Buat Space baru di Hugging Face (SDK: Gradio).
2. Upload isi folder `module_ML/hf_space/`.
3. Set environment variable `MODEL_REPO` ke repo model hasil fine-tuning (misal: `username/indobert-tokopedia-sentiment`).
4. Deploy.

Setelah aktif, Space akan menerima input ulasan dan menampilkan prediksi sentimen + skor probabilitas.

## Catatan Teknis

- Skrip preprocessing mencoba mendeteksi otomatis kolom teks dan label.
- Label akan dinormalisasi ke tiga kelas: `positif`, `netral`, `negatif`.
- Jika nama kolom berbeda dari default, gunakan argumen `--text-col` dan `--label-col` pada skrip training.
