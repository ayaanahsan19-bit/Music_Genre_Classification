"""
app.py — Gradio web demo for Music vs Speech Classification.
Deploy to HuggingFace Spaces: https://huggingface.co/spaces

Drag & drop a .wav file and get predictions from multiple models.
"""

import os
import warnings
import numpy as np

warnings.filterwarnings("ignore")

import gradio as gr

# ── Paths ───────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT, "outputs", "models")
CLASSES = ["music", "speech"]

SR = 22050
DURATION = 30
N_MFCC = 40
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
IMG_H, IMG_W = 224, 224


# ── Feature extraction (inline — no config dependency) ──────────────────────

def extract_features(wav_path: str) -> np.ndarray | None:
    """Extract audio features matching the tabular pipeline."""
    import librosa

    try:
        y, sr = librosa.load(wav_path, sr=SR, duration=DURATION)
    except Exception:
        return None

    features = []

    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    for c in mfcc:
        features.extend([np.mean(c), np.std(c)])

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    for c in chroma:
        features.extend([np.mean(c), np.std(c)])

    # Spectral features
    for feat_fn in [
        lambda: librosa.feature.spectral_centroid(y=y, sr=sr),
        lambda: librosa.feature.spectral_bandwidth(y=y, sr=sr),
        lambda: librosa.feature.spectral_rolloff(y=y, sr=sr),
        lambda: librosa.feature.zero_crossing_rate(y),
        lambda: librosa.feature.rms(y=y),
    ]:
        f = feat_fn()
        features.extend([np.mean(f), np.std(f)])

    # Tonnetz
    harm = librosa.effects.harmonic(y)
    tonnetz = librosa.feature.tonnetz(y=harm, sr=sr)
    features.append(np.mean(tonnetz))

    return np.array(features).reshape(1, -1)


# ── Tabular prediction ──────────────────────────────────────────────────────

def predict_tabular(wav_path: str) -> tuple[str, float] | None:
    import joblib

    for name in ("XGBoost.pkl", "RandomForest.pkl", "MLP.pkl"):
        p = os.path.join(MODELS_DIR, name)
        if os.path.exists(p):
            model = joblib.load(p)
            break
    else:
        return None

    feats = extract_features(wav_path)
    if feats is None:
        return None

    pred = model.predict(feats)[0]
    proba = model.predict_proba(feats)[0]
    return CLASSES[pred], float(np.max(proba))


# ── CNN prediction ──────────────────────────────────────────────────────────

def predict_cnn(wav_path: str) -> tuple[str, float] | None:
    try:
        import librosa
        import librosa.display
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import tensorflow as tf
    except ImportError:
        return None

    for ext in ("cnn_best.keras", "cnn_final.keras"):
        p = os.path.join(MODELS_DIR, ext)
        if os.path.exists(p):
            break
    else:
        return None

    y, sr = librosa.load(wav_path, sr=SR, duration=DURATION)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS,
                                       hop_length=HOP_LENGTH, n_fft=N_FFT)
    S_dB = librosa.power_to_db(S, ref=np.max)

    fig, ax = plt.subplots(1, 1, figsize=(IMG_W / 100, IMG_H / 100), dpi=100)
    librosa.display.specshow(S_dB, sr=sr, hop_length=HOP_LENGTH,
                             x_axis="time", y_axis="mel", cmap="viridis", ax=ax)
    ax.axis("off")
    fig.tight_layout(pad=0)
    tmp = os.path.join(ROOT, "_tmp_spec.png")
    fig.savefig(tmp, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    img = tf.keras.preprocessing.image.load_img(tmp, target_size=(IMG_H, IMG_W))
    x = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    model = tf.keras.models.load_model(p)
    proba = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(proba))
    os.remove(tmp)
    return CLASSES[idx], float(proba[idx])


# ── HuggingFace AST prediction ─────────────────────────────────────────────

def predict_hf(wav_path: str) -> tuple[str, float] | None:
    try:
        import torch
        import librosa
        from transformers import ASTForAudioClassification, ASTFeatureExtractor
    except ImportError:
        return None

    save_dir = os.path.join(MODELS_DIR, "hf_ast_best")
    if not os.path.isdir(save_dir):
        return None

    fe = ASTFeatureExtractor.from_pretrained(save_dir)
    model = ASTForAudioClassification.from_pretrained(save_dir)
    model.eval()

    model_sr = getattr(fe, "sampling_rate", 16000)
    y, _ = librosa.load(wav_path, sr=model_sr, duration=DURATION)
    max_len = model_sr * DURATION
    if len(y) < max_len:
        y = np.pad(y, (0, max_len - len(y)))
    else:
        y = y[:max_len]

    inputs = fe(y, sampling_rate=model_sr, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    proba = torch.softmax(logits, dim=-1).squeeze().numpy()
    idx = int(np.argmax(proba))
    return CLASSES[idx], float(proba[idx])


# ── Main classification function ────────────────────────────────────────────

def classify_audio(audio_path: str) -> dict:
    """Run all available models and return combined predictions."""
    if audio_path is None:
        return {"No audio uploaded": 0.0}

    results = {}

    # Tabular (best model — 100% accuracy)
    try:
        out = predict_tabular(audio_path)
        if out:
            label, conf = out
            results[f"Tabular → {label}"] = conf
    except Exception:
        pass

    # CNN
    try:
        out = predict_cnn(audio_path)
        if out:
            label, conf = out
            results[f"CNN → {label}"] = conf
    except Exception:
        pass

    # HuggingFace AST
    try:
        out = predict_hf(audio_path)
        if out:
            label, conf = out
            results[f"HF AST → {label}"] = conf
    except Exception:
        pass

    if not results:
        results["No models available"] = 0.0

    return results


# ── Gradio Interface ────────────────────────────────────────────────────────

demo = gr.Interface(
    fn=classify_audio,
    inputs=gr.Audio(type="filepath", label="Upload an audio file (.wav)"),
    outputs=gr.Label(num_top_classes=5, label="Predictions"),
    title="🎵  Music vs Speech Classifier",
    description=(
        "Upload an audio clip and see predictions from up to three models:\n\n"
        "**Tabular ML (XGBoost/RF)** · **Custom CNN** · **HuggingFace AST**\n\n"
        "Trained on the GTZAN Music/Speech dataset (128 clips)."
    ),
    article=(
        "### How it works\n"
        "- **Tabular:** Extracts 115 audio features (MFCCs, chroma, spectral) "
        "→ XGBoost/RandomForest classifier (100% test accuracy)\n"
        "- **CNN:** Converts audio to mel-spectrogram image → custom "
        "Conv2D network\n"
        "- **HF AST:** Audio Spectrogram Transformer with transfer learning "
        "from AudioSet (96.15% accuracy)\n\n"
        "[GitHub Repository](https://github.com/ayaanahsan19-bit/Music_Genre_Classification)"
    ),
    allow_flagging="never",
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch()
