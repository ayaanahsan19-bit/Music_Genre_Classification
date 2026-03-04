"""
predict.py — Single-file genre prediction CLI.

Usage:
    python src/inference/predict.py --file music_wav/sample.wav --model hf
    python src/inference/predict.py --file music_wav/sample.wav --model cnn
    python src/inference/predict.py --file music_wav/sample.wav --model tabular
"""

import os
import sys
import argparse
import yaml
import numpy as np
import warnings
warnings.filterwarnings("ignore")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

with open(os.path.join(ROOT, "config.yaml"), "r") as f:
    CFG = yaml.safe_load(f)

CLASSES = CFG["data"]["classes"]
SR = CFG["audio"]["sample_rate"]
DURATION = CFG["audio"]["duration"]
MODELS_DIR = os.path.join(ROOT, CFG["outputs"]["models_dir"])


# ── Tabular prediction ──────────────────────────────────────────────────────

def predict_tabular(wav_path: str) -> tuple[str, float]:
    """Predict genre using the saved tabular model (RandomForest by default)."""
    import joblib
    import librosa
    from src.preprocess.extract_features import extract_features_from_file

    feats = extract_features_from_file(wav_path)
    if feats is None:
        raise RuntimeError(f"Could not extract features from {wav_path}")

    # Find saved model
    for name in ("XGBoost.pkl", "MLP.pkl", "RandomForest.pkl"):
        p = os.path.join(MODELS_DIR, name)
        if os.path.exists(p):
            model_path = p
            break
    else:
        raise FileNotFoundError("No tabular model found in outputs/models/")

    model = joblib.load(model_path)

    # Build feature vector in same order (drop filename / label)
    feat_vals = np.array([v for k, v in feats.items()
                          if k not in ("filename", "label")]).reshape(1, -1)
    pred = model.predict(feat_vals)[0]
    proba = model.predict_proba(feat_vals)[0]
    conf = float(np.max(proba))
    return CLASSES[pred], conf


# ── CNN prediction ──────────────────────────────────────────────────────────

def predict_cnn(wav_path: str) -> tuple[str, float]:
    """Predict genre from the mel-spectrogram using the saved CNN."""
    import librosa
    import librosa.display
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from PIL import Image

    IMG_H = CFG["spectrogram"]["img_height"]
    IMG_W = CFG["spectrogram"]["img_width"]

    # Generate temp spectrogram
    y, sr = librosa.load(wav_path, sr=SR, duration=DURATION)
    S = librosa.feature.melspectrogram(y=y, sr=sr,
                                       n_mels=CFG["audio"]["n_mels"],
                                       hop_length=CFG["audio"]["hop_length"],
                                       n_fft=CFG["audio"]["n_fft"])
    S_dB = librosa.power_to_db(S, ref=np.max)
    fig, ax = plt.subplots(1, 1, figsize=(IMG_W / 100, IMG_H / 100), dpi=100)
    librosa.display.specshow(S_dB, sr=sr, hop_length=CFG["audio"]["hop_length"],
                             x_axis="time", y_axis="mel", cmap="viridis", ax=ax)
    ax.axis("off")
    fig.tight_layout(pad=0)
    tmp_path = os.path.join(ROOT, "outputs", "_tmp_spec.png")
    fig.savefig(tmp_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    # Load image
    img = tf.keras.preprocessing.image.load_img(tmp_path, target_size=(IMG_H, IMG_W))
    x = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    # Load CNN model
    for ext in ("cnn_best.keras", "cnn_final.keras"):
        p = os.path.join(MODELS_DIR, ext)
        if os.path.exists(p):
            model = tf.keras.models.load_model(p)
            break
    else:
        raise FileNotFoundError("No CNN model found in outputs/models/")

    proba = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(proba))
    os.remove(tmp_path)
    return CLASSES[idx], float(proba[idx])


# ── HuggingFace prediction ─────────────────────────────────────────────────

def predict_hf(wav_path: str) -> tuple[str, float]:
    """Predict genre using the fine-tuned HuggingFace AST model."""
    import torch
    import librosa
    from transformers import ASTForAudioClassification, ASTFeatureExtractor

    save_dir = os.path.join(MODELS_DIR, "hf_ast_best")
    if not os.path.isdir(save_dir):
        raise FileNotFoundError(f"HF model not found at {save_dir}")

    fe = ASTFeatureExtractor.from_pretrained(save_dir)
    model = ASTForAudioClassification.from_pretrained(save_dir)
    model.eval()

    # AST expects 16 kHz
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


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Music/Speech Prediction CLI")
    parser.add_argument("--file", required=True, help="Path to a .wav file")
    parser.add_argument("--model", default="hf",
                        choices=["tabular", "cnn", "hf"],
                        help="Which model to use (default: hf)")
    args = parser.parse_args()

    wav_path = os.path.abspath(args.file)
    if not os.path.isfile(wav_path):
        print(f"[ERROR] File not found: {wav_path}")
        sys.exit(1)

    dispatch = {"tabular": predict_tabular, "cnn": predict_cnn, "hf": predict_hf}
    fn = dispatch[args.model]

    print(f"  Model : {args.model}")
    print(f"  File  : {wav_path}")
    label, conf = fn(wav_path)
    print(f"\n  Predicted Class: {label}  (confidence: {conf:.2f})")


if __name__ == "__main__":
    main()
