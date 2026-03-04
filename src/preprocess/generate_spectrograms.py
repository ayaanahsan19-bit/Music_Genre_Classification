"""
generate_spectrograms.py — Convert .wav files to mel-spectrogram PNGs.

Reads from dataset/genres_original/ and saves 224×224 mel-spectrogram
images into dataset/images_original/<genre>/.
Also processes music_wav/ and speech_wav/ for inference.
"""

import os
import sys
import yaml
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")           # non-interactive backend
import matplotlib.pyplot as plt
from tqdm import tqdm

# ── Load config ──────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
with open(os.path.join(ROOT, "config.yaml"), "r") as f:
    CFG = yaml.safe_load(f)

DATASET_PATH = os.path.join(ROOT, CFG["data"]["dataset_path"])
SPEC_PATH = os.path.join(ROOT, CFG["data"]["spectrogram_path"])
MUSIC_WAV = os.path.join(ROOT, CFG["data"]["music_wav_path"])
SPEECH_WAV = os.path.join(ROOT, CFG["data"]["speech_wav_path"])
CLASSES = CFG["data"]["classes"]
SR = CFG["audio"]["sample_rate"]
DURATION = CFG["audio"]["duration"]
N_MELS = CFG["audio"]["n_mels"]
HOP = CFG["audio"]["hop_length"]
N_FFT = CFG["audio"]["n_fft"]
IMG_H = CFG["spectrogram"]["img_height"]
IMG_W = CFG["spectrogram"]["img_width"]


def wav_to_spectrogram(wav_path: str, out_path: str,
                       sr: int = SR, duration: int = DURATION) -> bool:
    """Convert a single .wav file to a mel-spectrogram PNG."""
    try:
        y, sr = librosa.load(wav_path, sr=sr, duration=duration)
    except Exception as e:
        print(f"  [SKIP] {wav_path}: {e}")
        return False

    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, hop_length=HOP, n_fft=N_FFT
    )
    S_dB = librosa.power_to_db(S, ref=np.max)

    # ── Render to image ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(IMG_W / 100, IMG_H / 100), dpi=100)
    librosa.display.specshow(S_dB, sr=sr, hop_length=HOP,
                             x_axis="time", y_axis="mel",
                             cmap="viridis", ax=ax)
    ax.axis("off")
    fig.tight_layout(pad=0)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return True


def process_class_folder(cls: str) -> int:
    """Generate spectrograms for all .wav files in a class folder."""
    src_dir = os.path.join(DATASET_PATH, cls)
    dst_dir = os.path.join(SPEC_PATH, cls)
    os.makedirs(dst_dir, exist_ok=True)

    if not os.path.isdir(src_dir):
        print(f"  [WARN] Source folder not found: {src_dir}")
        return 0

    wav_files = sorted([f for f in os.listdir(src_dir) if f.endswith(".wav")])
    count = 0
    for fname in tqdm(wav_files, desc=cls, leave=False):
        out_name = os.path.splitext(fname)[0] + ".png"
        out_path = os.path.join(dst_dir, out_name)
        if os.path.exists(out_path):
            count += 1
            continue
        if wav_to_spectrogram(os.path.join(src_dir, fname), out_path):
            count += 1
    return count


def process_custom_folder(folder_path: str, label: str) -> int:
    """Generate spectrograms for a custom .wav folder (music_wav / speech_wav)."""
    if not os.path.isdir(folder_path):
        return 0
    dst_dir = os.path.join(SPEC_PATH, label)
    os.makedirs(dst_dir, exist_ok=True)
    wav_files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
    count = 0
    for fname in tqdm(wav_files, desc=label, leave=False):
        out_name = os.path.splitext(fname)[0] + ".png"
        out_path = os.path.join(dst_dir, out_name)
        if os.path.exists(out_path):
            count += 1
            continue
        if wav_to_spectrogram(os.path.join(folder_path, fname), out_path):
            count += 1
    return count


def main():
    print("=" * 60)
    print("  Mel-Spectrogram Generation")
    print("=" * 60)
    total = 0
    for cls in CLASSES:
        n = process_class_folder(cls)
        print(f"  {cls}: {n} spectrograms")
        total += n
    print(f"\nSpectrograms generated: {total}")
    print("Done.")


if __name__ == "__main__":
    main()
