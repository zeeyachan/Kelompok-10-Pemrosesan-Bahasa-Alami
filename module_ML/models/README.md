## Model Storage Structure

This directory contains trained sentiment classification models:

### `/baseline/`
Traditional machine learning approach:
- **tfidf_logreg.joblib** - TF-IDF + Logistic Regression model (5.7 MB)
- **tfidf_svm.joblib** - TF-IDF + SVM model (5.7 MB)

**Performance**: 
- Accuracy: 94.36% ⭐
- Macro F1: 51.64%
- Inference: < 100ms (CPU only)

**Use case**: Production deployment, fast inference, limited resources

---

### `/deep_learning/`
Deep learning approach using transformer architecture:
- **final_model/** - Fine-tuned IndoBERT model ready for inference (475 MB)
- **checkpoints/** - Training checkpoints (optional, for resuming training)

**Performance**:
- Accuracy: 88.70%
- Macro F1: 50.88%
- Inference: ~500ms per sample

**Use case**: Better semantic understanding, research, explanation, nuanced sentiment

---

## Quick Load

```python
# Load baseline model
import joblib
baseline_model = joblib.load('models/baseline/tfidf_logreg.joblib')
predictions = baseline_model.predict(['teks review'])

# Load deep learning model
from transformers import AutoModelForSequenceClassification, AutoTokenizer
model = AutoModelForSequenceClassification.from_pretrained('models/deep_learning/final_model')
tokenizer = AutoTokenizer.from_pretrained('models/deep_learning/final_model')
inputs = tokenizer('teks review', return_tensors='pt')
outputs = model(**inputs)
```
