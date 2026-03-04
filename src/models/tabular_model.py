"""
tabular_model.py — Scikit-learn / XGBoost / MLP pipelines for tabular features.
"""

import os
import yaml
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
with open(os.path.join(ROOT, "config.yaml"), "r") as f:
    CFG = yaml.safe_load(f)

MODELS_DIR = os.path.join(ROOT, CFG["outputs"]["models_dir"])
os.makedirs(MODELS_DIR, exist_ok=True)


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_tabular_data(csv_path: str | None = None):
    """
    Load features CSV, split into X/y, encode labels, scale features.
    Falls back to the GTZAN features_30_sec.csv when no extracted CSV exists.
    """
    if csv_path is None:
        extracted = os.path.join(ROOT, "outputs", "features_extracted.csv")
        if os.path.exists(extracted):
            csv_path = extracted
        else:
            csv_path = os.path.join(ROOT, CFG["data"]["csv_path"])

    df = pd.read_csv(csv_path)

    # Drop non-feature columns
    drop_cols = [c for c in ("filename", "length") if c in df.columns]
    label_col = "label"
    X = df.drop(columns=drop_cols + [label_col])
    y = df[label_col]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    test_size = CFG["training"]["test_size"]
    rs = CFG["training"]["random_state"]
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=test_size, random_state=rs, stratify=y_enc
    )
    return X_train, X_test, y_train, y_test, le, scaler


# ── Model Builders ───────────────────────────────────────────────────────────

def build_random_forest(**kwargs) -> RandomForestClassifier:
    defaults = dict(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)
    defaults.update(kwargs)
    return RandomForestClassifier(**defaults)


def build_mlp(**kwargs) -> MLPClassifier:
    defaults = dict(
        hidden_layer_sizes=(512, 256, 128),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
    )
    defaults.update(kwargs)
    return MLPClassifier(**defaults)


def build_xgboost(**kwargs):
    if not HAS_XGB:
        raise ImportError("xgboost is not installed. pip install xgboost")
    defaults = dict(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
    )
    defaults.update(kwargs)
    return XGBClassifier(**defaults)


# ── Train / Evaluate helpers ────────────────────────────────────────────────

def train_and_evaluate(model, X_train, X_test, y_train, y_test, label_encoder,
                       model_name: str = "model"):
    """Fit, predict, print report, save model, return metrics dict."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=label_encoder.classes_, output_dict=True
    )
    print(f"\n{'─'*50}")
    print(f"  {model_name}  —  Accuracy: {acc:.4f}")
    print(f"{'─'*50}")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # Save
    save_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    joblib.dump(model, save_path)
    print(f"  Model saved → {save_path}")

    return {
        "model_name": model_name,
        "accuracy": acc,
        "f1_macro": report["macro avg"]["f1-score"],
        "y_test": y_test,
        "y_pred": y_pred,
    }
