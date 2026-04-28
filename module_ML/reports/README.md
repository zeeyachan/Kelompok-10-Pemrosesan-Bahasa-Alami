## Reports & Visualizations Guide

This directory contains all evaluation metrics, visualizations, and paper-ready materials.

---

## 📊 Visualization Files (PNG Images)

### 1. **metrics_comparison.png**
Bar chart perbandingan performance antara Baseline dan Deep Learning model.
- **X-axis**: Metrik (Accuracy, Macro F1, Weighted F1)
- **Y-axis**: Score (0-1)
- **Use**: Ideal untuk Figure 1 di paper/presentation
- **Size**: 125 KB

### 2. **confusion_matrices.png**
Side-by-side heatmap confusion matrices untuk kedua model.
- **Left**: Baseline model confusion matrix
- **Right**: Deep Learning model confusion matrix
- **Classes**: Negative (0), Neutral (1), Positive (2)
- **Use**: Untuk detail error analysis atau Figure 2
- **Size**: 134 KB

### 3. **model_summary.png**
Card-style visualization dengan model info, hyperparameters, dan key metrics.
- **Top section**: Model specifications
- **Bottom section**: Performance metrics
- **Use**: Quick reference untuk paper atau presentation
- **Size**: 225 KB

---

## 📄 JSON Report Files

### **arxiv_report.json** ⭐
**Main file untuk paper submission!**

Contains:
```json
{
  "paper_title": "...",
  "generated_at": "2026-04-28T...",
  "dataset": { /* Dataset info */ },
  "models": {
    "baseline": { /* TF-IDF + LogReg metrics */ },
    "deep_learning": { /* IndoBERT metrics */ }
  },
  "key_findings": [ /* Research insights */ ]
}
```

**Gunakan untuk:**
- Tabel metrik di paper
- Dataset description
- Model specifications
- Numerical results

### **baseline_logreg_metrics.json**
Detailed metrics untuk Baseline model:
- Algorithm: logreg
- Accuracy, Macro F1, Weighted F1
- Classification report per class
- Confusion matrix
- Test/Dataset size info

### **baseline_svm_metrics.json**
Performance metrics untuk SVM variant (experimental):
- Algorithm: svm
- Comparable metrics ke LogReg
- Useful untuk ablation study

### **transformer_metrics.json**
Detailed metrics untuk Deep Learning model:
- Model: indobenchmark/indobert-base-p1
- Training config (epochs, batch size, learning rate)
- Evaluation metrics
- Class distribution
- Sampling strategy info

---

## 🎯 How to View Files

### Option 1: View PNG Images in VS Code
1. Click on PNG filename in this folder
2. Images will open in preview pane
3. Right-click → "Open in External Editor" untuk melihat full size

### Option 2: View PNG from Command Line
```bash
# Linux/WSL
file metrics_comparison.png  # Check file info

# Or open with image viewer
eog metrics_comparison.png   # Eye of GNOME
display metrics_comparison.png  # ImageMagick
```

### Option 3: Mount Folder + View Locally
```bash
# Copy PNG to local machine
scp user@server:/path/reports/*.png ./local_folder/
```

### Option 4: Web Server Quick View
```bash
cd /path/to/reports
python -m http.server 8000
# Open browser: http://localhost:8000
```

---

## 📋 View JSON Files

### Option 1: Pretty Print di Terminal
```bash
# Install jq if not present
sudo apt-get install jq

# Pretty print
cat arxiv_report.json | jq .

# Query specific fields
cat arxiv_report.json | jq '.models.baseline'
```

### Option 2: Python Script
```python
import json

# Load report
with open('arxiv_report.json', 'r') as f:
    report = json.load(f)

# View pretty
print(json.dumps(report, indent=2, ensure_ascii=False))

# Access specific data
print("Baseline Accuracy:", report['models']['baseline']['metrics']['accuracy'])
```

### Option 3: VS Code JSON Viewer
1. Click on .json file
2. VS Code automatically formats dan bisa collapse/expand sections
3. Use Ctrl+F untuk search values

---

## 🚀 Using in Your Paper

### Copy Images to Paper
```bash
cp metrics_comparison.png /path/to/paper/figures/figure1.png
cp confusion_matrices.png /path/to/paper/figures/figure2.png
cp model_summary.png /path/to/paper/figures/figure3.png
```

### Extract Metrics for Tables
```python
import json

report = json.load(open('arxiv_report.json'))

# Create performance table
print("Model | Accuracy | Macro F1 | Weighted F1")
print("-" * 50)
for model_name, data in report['models'].items():
    m = data['metrics']
    print(f"{model_name:15} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | {m['weighted_f1']:.4f}")
```

### Create Captions
```md
**Figure 1**: Performance comparison between baseline (TF-IDF + LogReg) 
and deep learning (IndoBERT transformer) models on Tokopedia sentiment 
classification task. Baseline achieves higher accuracy (94.36%) with 
minimal computational cost.

**Figure 2**: Confusion matrices for both models showing detailed 
classification patterns. The baseline model shows better precision 
on positive class (top-right).

**Figure 3**: Model specifications and training parameters. Key 
differences: baseline trains in <30 seconds, deep learning requires 
30-60 minutes on GPU but provides better semantic understanding.
```

---

## 📊 File Sizes & Data

```
Total reports size: ~500 KB
├── PNG images: ~480 KB (high quality, 300 DPI)
├── JSON reports: ~20 KB (structured, human & machine readable)
└── Legacy metrics: Archived from previous runs

Trained models:
├── Baseline: 5.7 MB each (joblib format)
└── Deep Learning: 475 MB (safetensors format, optimized)
```

---

## ⚠️ Important Notes

1. **PNG Images** are high-resolution (300 DPI) suitable for publication
2. **JSON Files** are UTF-8 encoded, support Indonesian characters
3. **Metrics** are calculated on 80-20 stratified split with random_state=42
4. **Class Distribution**: Imbalanced - weighted metrics used accordingly
5. **Reproducibility**: All random seeds logged in json files

---

**Generated**: 2026-04-28 by automated training pipeline  
**Dataset**: Tokopedia Product Reviews 2025 (65,335 samples)  
**Language**: Indonesian  
**License**: Same as repository
