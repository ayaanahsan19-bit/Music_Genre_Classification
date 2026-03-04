"""
train_tabular.py — Train tabular classifiers (RF, MLP, XGBoost) on extracted features.
"""

import os
import sys
import time
import json
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Add project root to path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.models.tabular_model import (
    load_tabular_data,
    build_random_forest,
    build_mlp,
    build_xgboost,
    train_and_evaluate,
    HAS_XGB,
)

with open(os.path.join(ROOT, "config.yaml"), "r") as f:
    CFG = yaml.safe_load(f)

PLOTS_DIR = os.path.join(ROOT, CFG["outputs"]["plots_dir"])
RESULTS_DIR = os.path.join(ROOT, CFG["outputs"]["results_dir"])
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
CLASSES = CFG["data"]["classes"]


def plot_confusion_matrix(y_true, y_pred, labels, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved → {save_path}")


def main():
    print("=" * 60)
    print("  Tabular Model Training")
    print("=" * 60)

    X_train, X_test, y_train, y_test, le, scaler = load_tabular_data()
    print(f"  Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}  |  Features: {X_train.shape[1]}")

    results = []

    # ── Random Forest ───────────────────────────────────────────────────────
    print("\n▶ Training Random Forest...")
    t0 = time.time()
    rf = build_random_forest()
    rf_res = train_and_evaluate(rf, X_train, X_test, y_train, y_test, le, "RandomForest")
    rf_res["train_time"] = time.time() - t0
    results.append(rf_res)
    plot_confusion_matrix(
        rf_res["y_test"], rf_res["y_pred"], le.classes_,
        "Random Forest — Confusion Matrix",
        os.path.join(PLOTS_DIR, "cm_random_forest.png"),
    )

    # ── MLP ─────────────────────────────────────────────────────────────────
    print("\n▶ Training MLP...")
    t0 = time.time()
    mlp = build_mlp()
    mlp_res = train_and_evaluate(mlp, X_train, X_test, y_train, y_test, le, "MLP")
    mlp_res["train_time"] = time.time() - t0
    results.append(mlp_res)
    plot_confusion_matrix(
        mlp_res["y_test"], mlp_res["y_pred"], le.classes_,
        "MLP — Confusion Matrix",
        os.path.join(PLOTS_DIR, "cm_mlp.png"),
    )

    # ── XGBoost (optional) ──────────────────────────────────────────────────
    if HAS_XGB:
        print("\n▶ Training XGBoost...")
        t0 = time.time()
        xgb = build_xgboost()
        xgb_res = train_and_evaluate(xgb, X_train, X_test, y_train, y_test, le, "XGBoost")
        xgb_res["train_time"] = time.time() - t0
        results.append(xgb_res)
        plot_confusion_matrix(
            xgb_res["y_test"], xgb_res["y_pred"], le.classes_,
            "XGBoost — Confusion Matrix",
            os.path.join(PLOTS_DIR, "cm_xgboost.png"),
        )
    else:
        print("\n  [INFO] XGBoost not installed — skipping.")

    # ── Save summary ────────────────────────────────────────────────────────
    summary = []
    for r in results:
        summary.append({
            "model": r["model_name"],
            "accuracy": round(r["accuracy"], 4),
            "f1_macro": round(r["f1_macro"], 4),
            "train_time_s": round(r["train_time"], 2),
        })
    summary_path = os.path.join(RESULTS_DIR, "tabular_results.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved → {summary_path}")


if __name__ == "__main__":
    main()
