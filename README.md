# 🎵 Music vs Speech Classification

End-to-end audio classification system using the **GTZAN Music/Speech dataset** (64 music + 64 speech clips).  
Compares three approaches: **Tabular ML**, **Custom CNN on spectrograms**, and **HuggingFace Audio Spectrogram Transformer (AST)**.

---

## Results

| Approach | Accuracy | F1 Macro | Train Time |
|---|---|---|---|
| **Random Forest** | **100.00%** | **1.00** | 0.93 s |
| **XGBoost** | **100.00%** | **1.00** | 0.18 s |
| MLP | 96.15% | 0.96 | 0.58 s |
| HuggingFace AST | 96.15% | 0.96 | 1277 s |
| Custom CNN | 47.37% | 0.45 | 172 s |

> **Key takeaway:** Hand-crafted audio features (MFCCs, chroma, spectral) fed into tree-based models achieve perfect classification on this binary task. The pretrained AST transformer reaches 96% with only 3 epochs of head fine-tuning. The CNN trained from scratch overfits due to the small dataset size (128 images).

---

## Dataset

| Class | Samples | Source |
|---|---|---|
| Music | 64 | `music_wav/` → `dataset/genres_original/music/` |
| Speech | 64 | `speech_wav/` → `dataset/genres_original/speech/` |

Downloaded from [Kaggle — GTZAN Music and Speech](https://www.kaggle.com/datasets/lnicalo/gtzan-musicspeech-collection). Each clip is a 30-second mono `.wav` at 22 050 Hz.

---

## Project Structure

```
MUSIC_GENRE_CLASSIFICATION/
├── dataset/
│   ├── genres_original/          # .wav files organized by class
│   │   ├── music/                #   64 music clips
│   │   └── speech/               #   64 speech clips
│   └── images_original/          # Generated mel-spectrogram PNGs
│       ├── music/
│       └── speech/
├── music_wav/                    # Original music samples
├── speech_wav/                   # Original speech samples
├── src/
│   ├── preprocess/
│   │   ├── extract_features.py   # Librosa feature extraction → CSV
│   │   ├── generate_spectrograms.py  # Mel-spectrogram PNG generation
│   │   └── augment.py            # Time-stretch, pitch-shift, noise
│   ├── models/
│   │   ├── tabular_model.py      # RandomForest / MLP / XGBoost builders
│   │   ├── cnn_model.py          # Custom Keras CNN
│   │   └── hf_transfer.py        # HuggingFace AST transfer learning
│   ├── train/
│   │   ├── train_tabular.py      # Train & evaluate tabular classifiers
│   │   ├── train_cnn.py          # Train custom CNN on spectrograms
│   │   └── train_hf.py           # Fine-tune HuggingFace AST
│   ├── evaluate/
│   │   ├── metrics.py            # Shared metrics utilities
│   │   └── compare_approaches.py # Generate comparison report
│   └── inference/
│       ├── predict.py            # CLI single-file prediction
│       └── gradio_app.py         # Gradio web demo
├── notebooks/
│   └── eda.ipynb                 # Exploratory Data Analysis
├── outputs/
│   ├── models/                   # Saved model artifacts
│   │   ├── RandomForest.pkl
│   │   ├── MLP.pkl
│   │   ├── XGBoost.pkl
│   │   ├── cnn_best.keras
│   │   ├── cnn_final.keras
│   │   └── hf_ast_best/         # HuggingFace AST checkpoint
│   ├── plots/                    # Confusion matrices & learning curves
│   └── results/                  # JSON metrics & CSV comparison
│       ├── tabular_results.json
│       ├── cnn_results.json
│       ├── hf_results.json
│       └── comparison_report.csv
├── config.yaml                   # Central configuration
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/Music_Genre_Classification.git
cd Music_Genre_Classification
```

### 2. Create Virtual Environment (Python 3.11 recommended)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
pip install accelerate>=1.1.0
```

### 4. Prepare the Dataset

Download the [GTZAN Music/Speech Collection](https://www.kaggle.com/datasets/lnicalo/gtzan-musicspeech-collection) from Kaggle and place the files so the workspace contains:

```
music_wav/   ← 64 music .wav files
speech_wav/  ← 64 speech .wav files
```

The preprocessing scripts will automatically organize them into `dataset/genres_original/{music,speech}/`.

---

## Quick Start

Run each step sequentially:

```bash
# 1. Extract audio features (MFCCs, chroma, spectral) → outputs/features_extracted.csv
python src/preprocess/extract_features.py

# 2. Generate 224×224 mel-spectrogram PNGs → dataset/images_original/
python src/preprocess/generate_spectrograms.py

# 3. Train tabular classifiers (RF, MLP, XGBoost)
python src/train/train_tabular.py

# 4. Train custom CNN on spectrogram images
python src/train/train_cnn.py

# 5. Fine-tune HuggingFace AST (≈20 min on CPU)
python src/train/train_hf.py

# 6. Generate comparison report
python src/evaluate/compare_approaches.py
```

### Inference

```bash
# CLI — predict a single audio file
python src/inference/predict.py --file music_wav/music.00010.wav --model tabular
python src/inference/predict.py --file speech_wav/speech.00010.wav --model hf

# Gradio web demo
python src/inference/gradio_app.py
```

---

## Approach Details

### 1. Tabular Pipeline (scikit-learn + XGBoost)

Extracts **115 audio features** per clip using Librosa:

| Feature Group | Count | Description |
|---|---|---|
| MFCCs | 80 | 40 coefficients × mean + std |
| Chroma | 24 | 12 pitch classes × mean + std |
| Spectral Centroid | 2 | mean + std |
| Spectral Bandwidth | 2 | mean + std |
| Spectral Rolloff | 2 | mean + std |
| Zero-Crossing Rate | 2 | mean + std |
| RMS Energy | 2 | mean + std |
| Tonnetz | 1 | mean |

Three classifiers trained with `StandardScaler` preprocessing:
- **RandomForest** (100 estimators) → **100% accuracy**
- **XGBoost** (100 estimators) → **100% accuracy**
- **MLP** (256→128→64, ReLU, 500 epochs) → **96.15% accuracy**

### 2. Custom CNN (TensorFlow / Keras)

| Layer | Details |
|---|---|
| Input | 224 × 224 × 3 mel-spectrogram PNG |
| Conv2D blocks | 32 → 64 → 128 → 256 filters, 3×3 kernel, BatchNorm, MaxPool |
| Head | GlobalAvgPool → Dense(256) → Dropout(0.4) → Dense(2, softmax) |
| Params | 456,642 total (455,682 trainable) |

- Early stopping (patience=5) halted at epoch 18/30
- **47.37% test accuracy** — overfits on only 128 spectrogram images
- Best validation accuracy during training: 81.58% (epoch 8)

### 3. HuggingFace AST (Transfer Learning)

- **Base model:** [`MIT/ast-finetuned-audioset-10-10-0.4593`](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) (86M params)
- **Strategy:** Freeze all encoder layers, fine-tune only the 2-class classification head (3,074 params)
- **Audio preprocessing:** Resample to 16 kHz → AST feature extractor
- **Training:** 3 epochs via HuggingFace `Trainer` API
- **Result:** **96.15% accuracy** (1 misclassified sample out of 26)

---

## Configuration

All paths and hyperparameters are centralized in [`config.yaml`](config.yaml):

```yaml
data:
  classes: [music, speech]

audio:
  sample_rate: 22050
  n_mfcc: 40
  n_mels: 128

training:
  batch_size: 8
  epochs: 30
  learning_rate: 0.001

huggingface:
  model_name: "MIT/ast-finetuned-audioset-10-10-0.4593"
  num_labels: 2
  freeze_base: true
  fine_tune_epochs: 3
```

---

## Generated Outputs

After a full run, the `outputs/` directory contains:

| Directory | Contents |
|---|---|
| `outputs/models/` | `RandomForest.pkl`, `MLP.pkl`, `XGBoost.pkl`, `cnn_best.keras`, `cnn_final.keras`, `hf_ast_best/` |
| `outputs/plots/` | `cm_random_forest.png`, `cm_mlp.png`, `cm_xgboost.png`, `cm_cnn.png`, `cm_hf_ast.png`, `cnn_learning_curves.png` |
| `outputs/results/` | `tabular_results.json`, `cnn_results.json`, `hf_results.json`, `comparison_report.csv` |

---

## Requirements

- **Python 3.11** (TensorFlow does not yet support Python 3.13+)
- TensorFlow ≥ 2.13
- PyTorch
- HuggingFace Transformers ≥ 4.35, Accelerate ≥ 1.1.0
- Librosa 0.10.1, scikit-learn, XGBoost, Gradio

---

## License

For educational and research purposes.  
The GTZAN Music/Speech dataset is provided by [Marsyas / George Tzanetakis](http://marsyas.info/downloads/datasets.html).
