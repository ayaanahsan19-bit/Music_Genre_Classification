"""
train_hf.py — Fine-tune the HuggingFace Audio Spectrogram Transformer (AST)
on the GTZAN dataset for genre classification.
"""

import os
import sys
import json
import yaml
import time
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.models.hf_transfer import (
    build_ast_model,
    get_feature_extractor,
    build_file_list,
    GTZANAudioDataset,
    get_training_args,
    compute_metrics,
    GENRES,
    GENRE2IDX,
    IDX2GENRE,
)
from transformers import Trainer

with open(os.path.join(ROOT, "config.yaml"), "r") as f:
    CFG = yaml.safe_load(f)

MODELS_DIR = os.path.join(ROOT, CFG["outputs"]["models_dir"])
RESULTS_DIR = os.path.join(ROOT, CFG["outputs"]["results_dir"])
PLOTS_DIR = os.path.join(ROOT, CFG["outputs"]["plots_dir"])
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def main():
    print("=" * 60)
    print("  HuggingFace AST Fine-Tuning")
    print("=" * 60)

    # ── Build file list & split ─────────────────────────────────────────────
    files = build_file_list()
    if not files:
        print("[ERROR] No .wav files found in genre folders. "
              "Ensure dataset/genres_original/<genre>/*.wav exist.")
        sys.exit(1)

    train_files, test_files = train_test_split(
        files,
        test_size=CFG["training"]["test_size"],
        random_state=CFG["training"]["random_state"],
        stratify=[f[1] for f in files],
    )
    print(f"  Train: {len(train_files)}  |  Test: {len(test_files)}")

    # ── Feature extractor & datasets ────────────────────────────────────────
    fe = get_feature_extractor()
    train_ds = GTZANAudioDataset(train_files, fe)
    test_ds = GTZANAudioDataset(test_files, fe)

    # ── Model ───────────────────────────────────────────────────────────────
    model = build_ast_model()

    # Print trainable vs frozen parameter counts
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Parameters — Total: {total:,}  |  Trainable: {trainable:,}")

    # ── Trainer ─────────────────────────────────────────────────────────────
    training_args = get_training_args()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    # ── Train ───────────────────────────────────────────────────────────────
    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0
    print(f"\n  Training completed in {train_time:.1f}s")

    # ── Evaluate ────────────────────────────────────────────────────────────
    eval_results = trainer.evaluate()
    print(f"  Eval accuracy: {eval_results.get('eval_accuracy', 0):.4f}")
    print(f"  Eval F1 macro: {eval_results.get('eval_f1_macro', 0):.4f}")

    # ── Detailed predictions ────────────────────────────────────────────────
    preds_output = trainer.predict(test_ds)
    y_pred = np.argmax(preds_output.predictions, axis=-1)
    y_true = np.array([f[1] for f in test_files])
    print(classification_report(y_true, y_pred, target_names=GENRES))

    # ── Confusion matrix plot ───────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=GENRES, yticklabels=GENRES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("HuggingFace AST — Confusion Matrix")
    fig.tight_layout()
    cm_path = os.path.join(PLOTS_DIR, "cm_hf_ast.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved → {cm_path}")

    # ── Save model & results ────────────────────────────────────────────────
    save_dir = os.path.join(MODELS_DIR, "hf_ast_best")
    trainer.save_model(save_dir)
    fe.save_pretrained(save_dir)
    print(f"  Model saved → {save_dir}")

    summary = {
        "model": "HuggingFace_AST",
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "train_time_s": round(train_time, 2),
    }
    with open(os.path.join(RESULTS_DIR, "hf_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Results saved → {os.path.join(RESULTS_DIR, 'hf_results.json')}")


if __name__ == "__main__":
    main()
