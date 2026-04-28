# 📊 Kelompok 10 - Pemrosesan Bahasa Alami

Tugas Besar Mata Kuliah **Pemrosesan Bahasa Alami**  
Program Studi Sains Data — ITERA 2026

---

## 🎓 Full Training Results - Ready for ArXiv

**✨ All 4 models successfully trained on complete Tokopedia dataset (65,335 samples)**

### 🤖 Machine Learning Baselines (3 models) + 🧠 Deep Learning (1 model)

| Model | Accuracy | Macro F1 | Weighted F1 | Speed | Memory |
|-------|----------|----------|-------------|-------|--------|
| **1. TF-IDF + Logistic Regression** | 94.36% | 0.5164 | 0.9575 | ~10ms | ~500MB |
| **2. TF-IDF + Support Vector Machine** ⭐ ⭐ ⭐ | **97.60%** | **0.5506** | **0.9740** | ~15ms | ~500MB |
| **3. TF-IDF + Multinomial Naive Bayes** | 97.53% | 0.3292 | 0.9634 | **~5ms** | ~500MB |
| **4. IndoBERT Transformer** | 88.70% | 0.5088 | 0.9268 | ~500ms | ~500MB |

- 📖 [Full Module ML Documentation](module_ML/README.md)
- 📊 [Visualizations](module_ML/reports/) - metrics comparison, confusion matrices, model summary
- 📄 [ArXiv Report](module_ML/reports/arxiv_report.json) - paper-ready format

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

### 1️⃣ Baseline Machine Learning (3 Models)
**TF-IDF Feature Extraction + 3 Classifiers**

- **TF-IDF Vectorization**: 
  - Word n-grams: 1-3
  - Character n-grams: 2-4
  - Max features: 100,000
  - Normalization: L2

- **Classifiers**:
  1. **Logistic Regression** — 94.36% accuracy
  2. **Support Vector Machine (SVM)** — **97.60% accuracy ⭐ BEST ML**
  3. **Multinomial Naive Bayes** — 97.53% accuracy (fastest inference ~5ms)

### 2️⃣ Transformer Model (1 Model)
- **IndoBERT**: `indobenchmark/indobert-base-p1`  
- Menggunakan **weighted loss** untuk menangani imbalance data
- **88.70% accuracy** — best for semantic understanding

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
