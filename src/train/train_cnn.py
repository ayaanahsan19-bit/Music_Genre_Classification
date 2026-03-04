"""
train_cnn.py — Train the custom CNN on mel-spectrogram images.
"""

import os
import sys
import json
import yaml
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import tensorflow as tf
from src.models.cnn_model import build_cnn, get_data_generators

with open(os.path.join(ROOT, "config.yaml"), "r") as f:
    CFG = yaml.safe_load(f)

EPOCHS = CFG["training"]["epochs"]
MODELS_DIR = os.path.join(ROOT, CFG["outputs"]["models_dir"])
PLOTS_DIR = os.path.join(ROOT, CFG["outputs"]["plots_dir"])
RESULTS_DIR = os.path.join(ROOT, CFG["outputs"]["results_dir"])
CLASSES = CFG["data"]["classes"]
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def plot_history(history, save_path):
    """Plot training and validation accuracy / loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history["accuracy"], label="train")
    ax1.plot(history.history["val_accuracy"], label="val")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(history.history["loss"], label="train")
    ax2.plot(history.history["val_loss"], label="val")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Learning curves saved → {save_path}")


def plot_confusion_matrix(y_true, y_pred, labels, save_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Custom CNN — Confusion Matrix")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved → {save_path}")


def main():
    print("=" * 60)
    print("  Custom CNN Training (Keras)")
    print("=" * 60)

    # ── Data generators ─────────────────────────────────────────────────────
    train_gen, val_gen = get_data_generators()
    num_classes = train_gen.num_classes
    class_labels = list(train_gen.class_indices.keys())
    print(f"  Classes: {class_labels}")
    print(f"  Train samples: {train_gen.samples}  |  Val samples: {val_gen.samples}")

    # ── Build model ─────────────────────────────────────────────────────────
    model = build_cnn(num_classes=num_classes)
    model.summary()

    # ── Callbacks ───────────────────────────────────────────────────────────
    ckpt_path = os.path.join(MODELS_DIR, "cnn_best.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_accuracy", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=10, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
        ),
    ]

    # ── Train ───────────────────────────────────────────────────────────────
    t0 = time.time()
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1,
    )
    train_time = time.time() - t0

    # ── Plots ───────────────────────────────────────────────────────────────
    plot_history(history, os.path.join(PLOTS_DIR, "cnn_learning_curves.png"))

    # ── Evaluate on validation set ──────────────────────────────────────────
    val_gen.reset()
    y_pred_probs = model.predict(val_gen, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = val_gen.classes[:len(y_pred)]

    acc = np.mean(y_pred == y_true)
    report = classification_report(y_true, y_pred, target_names=class_labels,
                                   output_dict=True)
    print(f"\n  CNN Val Accuracy: {acc:.4f}")
    print(classification_report(y_true, y_pred, target_names=class_labels))

    plot_confusion_matrix(
        y_true, y_pred, class_labels,
        os.path.join(PLOTS_DIR, "cm_cnn.png"),
    )

    # ── Save final model & results ──────────────────────────────────────────
    final_path = os.path.join(MODELS_DIR, "cnn_final.keras")
    model.save(final_path)
    print(f"  Final model saved → {final_path}")

    summary = {
        "model": "CustomCNN",
        "accuracy": round(acc, 4),
        "f1_macro": round(report["macro avg"]["f1-score"], 4),
        "train_time_s": round(train_time, 2),
        "epochs_run": len(history.history["loss"]),
    }
    with open(os.path.join(RESULTS_DIR, "cnn_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Results saved → {os.path.join(RESULTS_DIR, 'cnn_results.json')}")


if __name__ == "__main__":
    main()
