"""
hf_transfer.py — HuggingFace Audio Spectrogram Transformer (AST)
for transfer learning on the GTZAN genre classification task.
"""

import os
import yaml
import torch
import numpy as np
import librosa
from torch.utils.data import Dataset, DataLoader
from transformers import (
    ASTForAudioClassification,
    ASTFeatureExtractor,
    TrainingArguments,
    Trainer,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
with open(os.path.join(ROOT, "config.yaml"), "r") as f:
    CFG = yaml.safe_load(f)

HF_MODEL = CFG["huggingface"]["model_name"]
NUM_LABELS = CFG["huggingface"]["num_labels"]
FREEZE_BASE = CFG["huggingface"]["freeze_base"]
FINE_TUNE_EPOCHS = CFG["huggingface"]["fine_tune_epochs"]
SR = CFG["audio"]["sample_rate"]
DURATION = CFG["audio"]["duration"]
CLASSES = CFG["data"]["classes"]
CLASS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLASS = {i: c for c, i in CLASS2IDX.items()}
# Keep GENRES as alias for backward compat in train_hf imports
GENRES = CLASSES
GENRE2IDX = CLASS2IDX
IDX2GENRE = IDX2CLASS
MODELS_DIR = os.path.join(ROOT, CFG["outputs"]["models_dir"])


# ── Dataset ──────────────────────────────────────────────────────────────────

class GTZANAudioDataset(Dataset):
    """Loads raw waveforms for the AST pipeline, resampling to model's expected SR."""

    def __init__(self, file_list: list[tuple[str, int]], feature_extractor,
                 sr: int = SR, duration: int = DURATION):
        self.files = file_list
        self.fe = feature_extractor
        # AST expects 16 kHz — use the feature extractor's sampling rate
        self.model_sr = getattr(feature_extractor, "sampling_rate", 16000)
        self.load_sr = sr
        self.duration = duration

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path, label = self.files[idx]
        # Load at model's expected sample rate
        y, _ = librosa.load(path, sr=self.model_sr, duration=self.duration)
        # Pad / truncate to fixed length
        max_len = self.model_sr * self.duration
        if len(y) < max_len:
            y = np.pad(y, (0, max_len - len(y)))
        else:
            y = y[:max_len]
        inputs = self.fe(y, sampling_rate=self.model_sr, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in inputs.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


# ── Build file list ──────────────────────────────────────────────────────────

def build_file_list(dataset_path: str | None = None):
    """Return list of (wav_path, label_idx) tuples."""
    if dataset_path is None:
        dataset_path = os.path.join(ROOT, CFG["data"]["dataset_path"])
    files = []
    for cls in CLASSES:
        cdir = os.path.join(dataset_path, cls)
        if not os.path.isdir(cdir):
            continue
        for f in sorted(os.listdir(cdir)):
            if f.endswith(".wav"):
                files.append((os.path.join(cdir, f), CLASS2IDX[cls]))
    return files


# ── Model builder ────────────────────────────────────────────────────────────

def build_ast_model(model_name=HF_MODEL, num_labels=NUM_LABELS,
                    freeze_base=FREEZE_BASE):
    """Load pre-trained AST and optionally freeze the base encoder."""
    model = ASTForAudioClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )
    if freeze_base:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
    return model


def get_feature_extractor(model_name=HF_MODEL):
    return ASTFeatureExtractor.from_pretrained(model_name)


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


# ── Training wrapper ────────────────────────────────────────────────────────

def get_training_args(output_dir: str | None = None):
    if output_dir is None:
        output_dir = os.path.join(MODELS_DIR, "hf_ast")
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=FINE_TUNE_EPOCHS,
        per_device_train_batch_size=CFG["training"]["batch_size"],
        per_device_eval_batch_size=CFG["training"]["batch_size"],
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        learning_rate=CFG["training"]["learning_rate"],
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )
