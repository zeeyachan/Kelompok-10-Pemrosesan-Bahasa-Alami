import os

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_REPO = os.getenv("MODEL_REPO", "w11wo/indonesian-roberta-base-sentiment-classifier")

# Label ini diasumsikan sama dengan label saat fine-tuning.
ID2LABEL = {0: "negatif", 1: "netral", 2: "positif"}


def load_model(model_repo: str):
    tokenizer = AutoTokenizer.from_pretrained(model_repo)
    model = AutoModelForSequenceClassification.from_pretrained(model_repo)
    model.eval()
    return tokenizer, model


def predict_sentiment(text: str):
    if not text or not text.strip():
        return "Input kosong", {"negatif": 0.0, "netral": 0.0, "positif": 0.0}

    encoded = TOKENIZER(
        text.strip(),
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )

    with torch.no_grad():
        logits = MODEL(**encoded).logits
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy().tolist()

    labels = [ID2LABEL.get(i, str(i)) for i in range(len(probs))]
    score_map = {label: float(score) for label, score in zip(labels, probs)}
    pred_label = max(score_map, key=score_map.get)

    return pred_label, score_map


def bootstrap_model():
    try:
        return load_model(MODEL_REPO)
    except Exception:
        # Fallback agar Space tetap bisa berjalan jika model custom belum tersedia.
        fallback_repo = "w11wo/indonesian-roberta-base-sentiment-classifier"
        return load_model(fallback_repo)


TOKENIZER, MODEL = bootstrap_model()


demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=5, label="Masukkan ulasan produk Tokopedia"),
    outputs=[
        gr.Label(label="Prediksi Sentimen"),
        gr.Label(label="Probabilitas Kelas"),
    ],
    title="Analisis Sentimen Ulasan Tokopedia",
    description=(
        "Model IndoBERT fine-tuned untuk klasifikasi sentimen: positif, netral, negatif."
    ),
    examples=[
        ["Barang cepat sampai dan kualitas sesuai, puas sekali!"],
        ["Produk lumayan, packaging oke, tapi pengiriman agak lambat."],
        ["Barang rusak saat diterima, sangat mengecewakan."],
    ],
    allow_flagging="never",
)


if __name__ == "__main__":
    demo.launch()
