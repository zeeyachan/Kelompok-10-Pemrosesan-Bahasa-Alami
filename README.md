# 📊 Kelompok 10 - Pemrosesan Bahasa Alami

Tugas Besar Mata Kuliah **Pemrosesan Bahasa Alami**  
Program Studi Sains Data — ITERA 2026

---

## 👥 Anggota Tim

- <img src="https://github.com/zeeyachan.png" width="22"> **Nabila Zakiyah Zahra** (122450139) — [@zeeyachan](https://github.com/zeeyachan)
- <img src="https://github.com/salwaf01.png" width="22"> **Salwa Farhanatussaidah** (122450055) — [@salwaf01](https://github.com/salwaf01)
- <img src="https://github.com/nasywanaff.png" width="22"> **Nasywa Nur Afifah** (122450125) — [@nasywanaff](https://github.com/nasywanaff)

---

## 📂 Dataset

Dataset yang digunakan berasal dari Kaggle:  
https://www.kaggle.com/datasets/salmanabdu/tokopedia-product-reviews-2025  

Dataset ini berisi **ulasan produk Tokopedia** yang digunakan untuk analisis sentimen dalam Bahasa Indonesia.

### 📝 Deskripsi
- **Jenis data**: teks ulasan pengguna  
- **Bahasa**: Indonesia  
- **Domain**: e-commerce  
- **Tujuan**: klasifikasi sentimen  

### 🧱 Struktur Data
Kolom utama pada dataset:
- `review_text` → isi ulasan pengguna (**kolom teks**)  
- `rating` → nilai rating (1–5)  
- `label` → kategori sentimen (**kolom target**)  

### 🎯 Label Sentimen
- `positif`  
- `netral`  
- `negatif`  

### 📊 Contoh Data

| review_text | rating | label |
|------------|--------|-------|
| Barang bagus, sesuai deskripsi | 5 | positif |
| Pengiriman lama, barang kurang rapi | 2 | negatif |

### ⚠️ Catatan
- Data berupa teks tidak formal (mengandung typo, singkatan, dll)  
- Dilakukan preprocessing sebelum digunakan dalam model  

---

## 🎯 Tujuan Proyek

Membangun sistem **analisis sentimen ulasan produk Tokopedia** untuk mengklasifikasikan teks ke dalam:

- Positif  
- Netral  
- Negatif  

---

## ⚙️ Pendekatan

Pendekatan yang digunakan dalam proyek ini:

### 1. Baseline Machine Learning
- TF-IDF  
- Logistic Regression  
- Support Vector Machine (SVM)  

### 2. Transformer Model
- IndoBERT: `indobenchmark/indobert-base-p1`  
- Menggunakan **weighted loss** untuk menangani imbalance data  

### 📈 Evaluasi
Model dibandingkan menggunakan:
- Accuracy  
- Macro F1-score  
- Weighted F1-score  

---

## 📁 Struktur Proyek

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
