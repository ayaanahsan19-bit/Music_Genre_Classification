"""
extract_features.py — Librosa-based audio feature extraction.

Extracts MFCCs, Chroma STFT, Spectral features, ZCR, and RMS
from every .wav in dataset/genres_original/ and saves to CSV.
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

# ── Load config ──────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
with open(os.path.join(ROOT, "config.yaml"), "r") as f:
    CFG = yaml.safe_load(f)

DATASET_PATH = os.path.join(ROOT, CFG["data"]["dataset_path"])
CLASSES = CFG["data"]["classes"]
SR = CFG["audio"]["sample_rate"]
DURATION = CFG["audio"]["duration"]
N_MFCC = CFG["audio"]["n_mfcc"]
N_MELS = CFG["audio"]["n_mels"]
HOP = CFG["audio"]["hop_length"]
N_FFT = CFG["audio"]["n_fft"]
OUTPUT_CSV = os.path.join(ROOT, "outputs", "features_extracted.csv")


def extract_features_from_file(file_path: str) -> dict | None:
    """Extract a comprehensive feature vector from a single audio file."""
    try:
        y, sr = librosa.load(file_path, sr=SR, duration=DURATION)
    except Exception as e:
        print(f"  [SKIP] {file_path}: {e}")
        return None

    features: dict = {}

    # ── MFCCs (mean + std of each coefficient → 2 × n_mfcc features) ────────
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP, n_fft=N_FFT)
    for i in range(N_MFCC):
        features[f"mfcc{i+1}_mean"] = np.mean(mfccs[i])
        features[f"mfcc{i+1}_std"] = np.std(mfccs[i])

    # ── Chroma STFT (12 features) ───────────────────────────────────────────
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP, n_fft=N_FFT)
    for i in range(12):
        features[f"chroma{i+1}_mean"] = np.mean(chroma[i])

    # ── Spectral Centroid ────────────────────────────────────────────────────
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP, n_fft=N_FFT)
    features["spectral_centroid_mean"] = np.mean(spec_cent)
    features["spectral_centroid_std"] = np.std(spec_cent)

    # ── Spectral Bandwidth ──────────────────────────────────────────────────
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=HOP, n_fft=N_FFT)
    features["spectral_bandwidth_mean"] = np.mean(spec_bw)
    features["spectral_bandwidth_std"] = np.std(spec_bw)

    # ── Spectral Rolloff ────────────────────────────────────────────────────
    spec_ro = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=HOP, n_fft=N_FFT)
    features["spectral_rolloff_mean"] = np.mean(spec_ro)
    features["spectral_rolloff_std"] = np.std(spec_ro)

    # ── Zero Crossing Rate ──────────────────────────────────────────────────
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP)
    features["zcr_mean"] = np.mean(zcr)
    features["zcr_std"] = np.std(zcr)

    # ── RMS Energy ──────────────────────────────────────────────────────────
    rms = librosa.feature.rms(y=y, hop_length=HOP)
    features["rms_mean"] = np.mean(rms)
    features["rms_std"] = np.std(rms)

    # ── Spectral Contrast (7 bands) ────────────────────────────────────────
    spec_con = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=HOP, n_fft=N_FFT)
    for i in range(7):
        features[f"spectral_contrast{i+1}_mean"] = np.mean(spec_con[i])

    # ── Tonnetz (6 features) ────────────────────────────────────────────────
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    for i in range(6):
        features[f"tonnetz{i+1}_mean"] = np.mean(tonnetz[i])

    return features


def extract_all(dataset_path: str = DATASET_PATH, classes: list = CLASSES) -> pd.DataFrame:
    """Walk class folders and build a feature DataFrame."""
    rows = []
    for cls in classes:
        cls_dir = os.path.join(dataset_path, cls)
        if not os.path.isdir(cls_dir):
            print(f"  [WARN] Class folder not found: {cls_dir}")
            continue
        wav_files = sorted([f for f in os.listdir(cls_dir) if f.endswith(".wav")])
        print(f"Processing {cls} ({len(wav_files)} files)...")
        for fname in tqdm(wav_files, desc=cls, leave=False):
            fpath = os.path.join(cls_dir, fname)
            feats = extract_features_from_file(fpath)
            if feats is not None:
                feats["filename"] = fname
                feats["label"] = cls
                rows.append(feats)
    df = pd.DataFrame(rows)
    return df


def main():
    print("=" * 60)
    print("  Music/Speech Feature Extraction")
    print("=" * 60)
    df = extract_all()
    if df.empty:
        print("\n[ERROR] No features extracted. Check that dataset/genres_original/ "
              "contains class sub-folders (music/, speech/) with .wav files.")
        sys.exit(1)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(df)} rows × {len(df.columns)} cols → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
