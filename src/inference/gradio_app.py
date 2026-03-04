"""
gradio_app.py — Gradio web demo for Music Genre Classification.

Drag-and-drop a .wav file and get genre predictions from all three models.
Run:  python src/inference/gradio_app.py
"""

import os
import sys
import yaml
import warnings
warnings.filterwarnings("ignore")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

with open(os.path.join(ROOT, "config.yaml"), "r") as f:
    CFG = yaml.safe_load(f)

CLASSES = CFG["data"]["classes"]

import gradio as gr


def classify_audio(audio_path: str) -> dict:
    """Run all available models and return a combined prediction dict."""
    from src.inference.predict import predict_tabular, predict_cnn, predict_hf

    results = {}

    # ── Tabular ─────────────────────────────────────────────────────────────
    try:
        genre, conf = predict_tabular(audio_path)
        results[f"Tabular → {genre}"] = conf
    except Exception as e:
        results["Tabular → (error)"] = 0.0

    # ── CNN ─────────────────────────────────────────────────────────────────
    try:
        genre, conf = predict_cnn(audio_path)
        results[f"CNN → {genre}"] = conf
    except Exception as e:
        results["CNN → (error)"] = 0.0

    # ── HuggingFace AST ────────────────────────────────────────────────────
    try:
        genre, conf = predict_hf(audio_path)
        results[f"HF AST → {genre}"] = conf
    except Exception as e:
        results["HF AST → (error)"] = 0.0

    return results


demo = gr.Interface(
    fn=classify_audio,
    inputs=gr.Audio(type="filepath", label="Upload a .wav audio file"),
    outputs=gr.Label(num_top_classes=5, label="Genre Predictions"),
    title="🎵  Music Genre Classifier",
    description=(
        "Upload a music clip (.wav) and see predictions from three models:\n"
        "**Tabular (RF/MLP)** · **Custom CNN** · **HuggingFace AST**"
    ),
    examples=[],
    allow_flagging="never",
)


if __name__ == "__main__":
    demo.launch(share=False)
