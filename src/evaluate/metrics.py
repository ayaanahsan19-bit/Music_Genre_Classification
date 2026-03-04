"""
metrics.py — Shared evaluation utilities: accuracy, F1, confusion matrix helpers.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def compute_all_metrics(y_true, y_pred, label_names=None) -> dict:
    """Return a dict with accuracy, f1_macro, and per-class report."""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, target_names=label_names,
                                   output_dict=True)
    return {"accuracy": acc, "f1_macro": f1, "report": report}


def print_report(y_true, y_pred, label_names=None, title="Evaluation"):
    """Pretty-print classification metrics."""
    print(f"\n{'═'*50}")
    print(f"  {title}")
    print(f"{'═'*50}")
    print(f"  Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"  F1 macro : {f1_score(y_true, y_pred, average='macro'):.4f}")
    print()
    print(classification_report(y_true, y_pred, target_names=label_names))


def save_confusion_matrix(y_true, y_pred, label_names, title, save_path):
    """Plot and save a confusion-matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path
