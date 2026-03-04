"""
compare_approaches.py — Load results from all three pipelines and produce
a side-by-side comparison table + CSV report.
"""

import os
import sys
import json
import yaml
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

with open(os.path.join(ROOT, "config.yaml"), "r") as f:
    CFG = yaml.safe_load(f)

RESULTS_DIR = os.path.join(ROOT, CFG["outputs"]["results_dir"])


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def main():
    print("=" * 60)
    print("  Approach Comparison")
    print("=" * 60)

    rows = []

    # ── Tabular results ─────────────────────────────────────────────────────
    tab_path = os.path.join(RESULTS_DIR, "tabular_results.json")
    tab_data = load_json(tab_path)
    if tab_data:
        if isinstance(tab_data, list):
            rows.extend(tab_data)
        else:
            rows.append(tab_data)
    else:
        print("  [WARN] tabular_results.json not found — run train_tabular.py first")

    # ── CNN results ─────────────────────────────────────────────────────────
    cnn_path = os.path.join(RESULTS_DIR, "cnn_results.json")
    cnn_data = load_json(cnn_path)
    if cnn_data:
        rows.append(cnn_data)
    else:
        print("  [WARN] cnn_results.json not found — run train_cnn.py first")

    # ── HF AST results ──────────────────────────────────────────────────────
    hf_path = os.path.join(RESULTS_DIR, "hf_results.json")
    hf_data = load_json(hf_path)
    if hf_data:
        rows.append(hf_data)
    else:
        print("  [WARN] hf_results.json not found — run train_hf.py first")

    if not rows:
        print("\n  No results found. Train at least one model first.")
        sys.exit(1)

    # ── Build comparison DataFrame ──────────────────────────────────────────
    df = pd.DataFrame(rows)
    cols_order = [c for c in ["model", "accuracy", "f1_macro", "train_time_s", "epochs_run"]
                  if c in df.columns]
    df = df[cols_order]
    df = df.sort_values("accuracy", ascending=False).reset_index(drop=True)

    print("\n" + df.to_string(index=False))

    # ── Save CSV ────────────────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, "comparison_report.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Comparison saved → {csv_path}")

    # ── Highlight best ──────────────────────────────────────────────────────
    best = df.iloc[0]
    print(f"\n  🏆 Best model: {best['model']}  "
          f"(Acc={best['accuracy']:.4f}, F1={best['f1_macro']:.4f})")


if __name__ == "__main__":
    main()
