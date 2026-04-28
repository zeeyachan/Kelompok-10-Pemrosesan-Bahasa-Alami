# 🚀 PANDUAN UPLOAD MODELS KE HUGGING FACE

Dokumentasi lengkap untuk upload model ML & DL ke Hugging Face Hub.

## 📋 Persyaratan

```bash
# Install required packages
pip install huggingface-hub
pip install huggingface_hub[cli]
```

## 🔐 Setup Authentication

### Opsi 1: Menggunakan HF CLI (Recommended)

```bash
# Login ke akun Hugging Face Anda
huggingface-cli login

# Masukkan token Anda (get from: https://huggingface.co/settings/tokens)
# Token kind: write access
```

### Opsi 2: Menggunakan Environment Variable

```bash
export HF_TOKEN="your_huggingface_token_here"
```

### Opsi 3: Menggunakan Python

```python
from huggingface_hub import login
login(token="your_token_here")
```

## 📦 Struktur Upload

Deployments akan dibuat di organization Anda:

```
https://huggingface.co/kelompok-10-NLP-SD-2026/
├── indobert-tokopedia-sentiment/     (Deep Learning Model)
├── tfidf-sentiment-baseline/         (ML Baseline Models)
└── tokopedia-sentiment-classifier/   (Gradio Space - optional)
```

## 🎯 Cara Upload

### Upload SEMUA Models + Space

```bash
cd /workspaces/pba2026-kelompok10

python module_ML/deploy_hf.py \
  --org kelompok-10-NLP-SD-2026 \
  --upload-dl \
  --upload-ml \
  --upload-space
```

### Upload HANYA Deep Learning Model

```bash
python module_ML/deploy_hf.py \
  --org kelompok-10-NLP-SD-2026 \
  --upload-dl
```

### Upload HANYA ML Models

```bash
python module_ML/deploy_hf.py \
  --org kelompok-10-NLP-SD-2026 \
  --upload-ml
```

### Upload HANYA Space

```bash
python module_ML/deploy_hf.py \
  --org kelompok-10-NLP-SD-2026 \
  --upload-space
```

## 🔧 Advanced Options

### Custom Paths

```bash
python module_ML/deploy_hf.py \
  --org kelompok-10-NLP-SD-2026 \
  --upload-dl \
  --dl-dir /custom/path/to/model
```

### Full Example dengan Custom Paths

```bash
python module_ML/deploy_hf.py \
  --org kelompok-10-NLP-SD-2026 \
  --upload-dl \
  --upload-ml \
  --upload-space \
  --dl-dir module_ML/models/deep_learning/final_model \
  --ml-dir module_ML/models/baseline \
  --space-dir module_ML/hf_space
```

## 📊 Output Setelah Upload

Setelah suksesnya, Anda akan mendapatkan:

```
======================================================================
🤖 UPLOAD MODELS KE HUGGING FACE
======================================================================

⏳ Uploading IndoBERT model to kelompok-10-NLP-SD-2026/indobert-tokopedia-sentiment...
✅ IndoBERT model uploaded: https://huggingface.co/kelompok-10-NLP-SD-2026/indobert-tokopedia-sentiment

⏳ Uploading ML baseline models to kelompok-10-NLP-SD-2026/tfidf-sentiment-baseline...
✅ ML models uploaded: https://huggingface.co/kelompok-10-NLP-SD-2026/tfidf-sentiment-baseline

⏳ Uploading Space app to kelompok-10-NLP-SD-2026/tokopedia-sentiment-classifier...
✅ Space app updated: https://huggingface.co/spaces/kelompok-10-NLP-SD-2026/tokopedia-sentiment-classifier

======================================================================
✅ UPLOAD SELESAI!
======================================================================

📦 Models tersedia di: https://huggingface.co/kelompok-10-NLP-SD-2026
   DL Model:  https://huggingface.co/kelompok-10-NLP-SD-2026/indobert-tokopedia-sentiment
   ML Model:  https://huggingface.co/kelompok-10-NLP-SD-2026/tfidf-sentiment-baseline
   Space:     https://huggingface.co/spaces/kelompok-10-NLP-SD-2026/tokopedia-sentiment-classifier
```

## 💻 Cara Download & Gunakan dari HF

### Download Dari HF Hub

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import joblib

# Deep Learning Model
print("Loading IndoBERT model...")
dl_model = AutoModelForSequenceClassification.from_pretrained(
    "kelompok-10-NLP-SD-2026/indobert-tokopedia-sentiment"
)
dl_tokenizer = AutoTokenizer.from_pretrained(
    "kelompok-10-NLP-SD-2026/indobert-tokopedia-sentiment"
)

# ML Baseline Models
print("Loading ML baseline models...")
ml_logreg = joblib.load('tfidf_logreg.joblib')  # If downloaded locally
```

### Make Predictions dengan DL Model

```python
text = "Produk sangat bagus dan rekomendasi!"

# Tokenize
inputs = dl_tokenizer(text, return_tensors="pt")

# Predict
outputs = dl_model(**inputs)
predicted_class = outputs.logits.argmax(-1).item()
confidence = outputs.logits.softmax(dim=-1).max().item()

labels = {0: "Negatif", 1: "Netral", 2: "Positif"}
print(f"Sentiment: {labels[predicted_class]}")
print(f"Confidence: {confidence:.2%}")
```

### Make Predictions dengan ML Model

```python
import joblib

# Load dari HF Hub (atau locally)
model = joblib.load('tfidf_logreg.joblib')

# Predict
review = "Barang rusak, tidak puas"
prediction = model.predict([review])
probabilities = model.predict_proba([review])

labels = {0: "Negatif", 1: "Netral", 2: "Positif"}
print(f"Sentiment: {labels[prediction[0]]}")
print(f"Probabilities: {probabilities[0]}")
```

## 📱 Akses Repository

Setelah upload:

1. **Visit HF Hub**: https://huggingface.co/kelompok-10-NLP-SD-2026
2. **View Model Card**: Automatically generated dari README.md
3. **Model Files**: Semua files dapat di-browse
4. **Clone Models**:
   ```bash
   git clone https://huggingface.co/kelompok-10-NLP-SD-2026/indobert-tokopedia-sentiment
   git clone https://huggingface.co/kelompok-10-NLP-SD-2026/tfidf-sentiment-baseline
   ```

## 🔗 Integration dengan Paper

### Cite dalam Paper

```bibtex
@misc{kelompok10_sentiment_2026,
  title={Indonesian E-commerce Sentiment Classification},
  author={Kelompok 10},
  year={2026},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/kelompok-10-NLP-SD-2026}}
}
```

### Model Card Links

```markdown
**Deep Learning Model**: https://huggingface.co/kelompok-10-NLP-SD-2026/indobert-tokopedia-sentiment
**Baseline Models**: https://huggingface.co/kelompok-10-NLP-SD-2026/tfidf-sentiment-baseline
```

## 🐛 Troubleshooting

### Error: "Token is invalid"
```bash
# Re-login
huggingface-cli logout
huggingface-cli login
# Enter token from https://huggingface.co/settings/tokens
```

### Error: "Failed to upload"
```bash
# Check file permissions
ls -la module_ML/models/

# Try with verbose output
python module_ML/deploy_hf.py \
  --org kelompok-10-NLP-SD-2026 \
  --upload-dl \
  --upload-ml \
  --upload-space 2>&1 | tee upload.log
```

### Error: "Organization not found"
- Pastikan organization sudah dibuat di Hugging Face
- Verify URL: https://huggingface.co/kelompok-10-NLP-SD-2026
- Ensure Anda memiliki write access ke organization

### Model terlalu besar?
```bash
# Check model sizes
du -sh module_ML/models/baseline/*
du -sh module_ML/models/deep_learning/*

# Large file storage (LFS) akan digunakan otomatis untuk file > 5GB
```

## ✅ Verification Checklist

Setelah upload, verify:

- [ ] Models muncul di organization page
- [ ] Model card terlihat dengan proper formatting
- [ ] Dapat download models dari HF Hub
- [ ] Inference berfungsi dengan baik
- [ ] Links dapat di-share
- [ ] README lengkap dan informative

## 🎯 Next Steps

1. **Share Models**: Bagikan links ke colleques
2. **Document**: Update paper dengan model links
3. **Cite**: Include dalam references
4. **Deploy**: Gunakan di applications
5. **Monitor**: Track downloads & engagement

## 📚 Resources

- **HF Docs**: https://huggingface.co/docs
- **HF Hub Guide**: https://huggingface.co/docs/hub/
- **Model Card**: https://huggingface.co/docs/hub/model-cards
- **Transformers**: https://huggingface.co/docs/transformers

---

**Questions?** Check GitHub issues atau baca dokumentasi di repo!
