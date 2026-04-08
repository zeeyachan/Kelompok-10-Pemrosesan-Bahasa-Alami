import os

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_REPO = os.getenv("MODEL_REPO", "indobenchmark/indobert-base-p1")
FALLBACK_REPO = "indobenchmark/indobert-base-p1"
DEFAULT_ID2LABEL = {0: "negatif", 1: "netral", 2: "positif"}


def resolve_id2label(model):
    raw_id2label = getattr(model.config, "id2label", None) or {}
    resolved = {}
    for idx, default_label in DEFAULT_ID2LABEL.items():
        raw_label = raw_id2label.get(idx)
        if raw_label is None:
            raw_label = raw_id2label.get(str(idx))
        if isinstance(raw_label, str) and not raw_label.startswith("LABEL_"):
            resolved[idx] = raw_label
        else:
            resolved[idx] = default_label
    return resolved

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

    id2label = resolve_id2label(MODEL)
    labels = [id2label.get(i, str(i)) for i in range(len(probs))]
    score_map = {label: float(score) for label, score in zip(labels, probs)}
    pred_label = max(score_map, key=score_map.get)

    return pred_label, score_map


def bootstrap_model():
    try:
        return load_model(MODEL_REPO)
    except Exception:
        # Fallback agar Space tetap bisa berjalan untuk smoke test jika repo custom belum tersedia.
        return load_model(FALLBACK_REPO)


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
