"""
cnn_model.py — Custom CNN (Keras / TensorFlow) for spectrogram classification.
"""

import os
import yaml
import tensorflow as tf
from tensorflow import keras
from keras import layers, models

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
with open(os.path.join(ROOT, "config.yaml"), "r") as f:
    CFG = yaml.safe_load(f)

IMG_H = CFG["spectrogram"]["img_height"]
IMG_W = CFG["spectrogram"]["img_width"]
NUM_CLASSES = len(CFG["data"]["classes"])
LR = CFG["training"]["learning_rate"]


def build_cnn(input_shape=(IMG_H, IMG_W, 3), num_classes=NUM_CLASSES,
              learning_rate=LR) -> keras.Model:
    """
    Custom CNN for mel-spectrogram images.

    Architecture:
        Input(224,224,3)
        → Conv2D(32,3) → BN → ReLU → MaxPool(2)
        → Conv2D(64,3) → BN → ReLU → MaxPool(2)
        → Conv2D(128,3) → BN → ReLU → MaxPool(2)
        → Conv2D(256,3) → BN → ReLU → GlobalAvgPool
        → Dense(256) → Dropout(0.4)
        → Dense(num_classes, softmax)
    """
    inp = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, (3, 3), padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(256, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)

    out = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=inp, outputs=out, name="GenreCNN")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_data_generators(spectrogram_dir: str | None = None,
                        batch_size: int | None = None,
                        val_split: float | None = None,
                        seed: int | None = None):
    """
    Return (train_gen, val_gen) using Keras ImageDataGenerator.
    """
    if spectrogram_dir is None:
        spectrogram_dir = os.path.join(ROOT, CFG["data"]["spectrogram_path"])
    if batch_size is None:
        batch_size = CFG["training"]["batch_size"]
    if val_split is None:
        val_split = CFG["training"]["val_size"] + CFG["training"]["test_size"]
    if seed is None:
        seed = CFG["training"]["random_state"]

    datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=val_split,
        rotation_range=5,
        zoom_range=0.1,
        width_shift_range=0.05,
        height_shift_range=0.05,
        brightness_range=(0.8, 1.2),
    )

    common = dict(
        directory=spectrogram_dir,
        target_size=(IMG_H, IMG_W),
        batch_size=batch_size,
        class_mode="categorical",
        seed=seed,
        shuffle=True,
    )

    train_gen = datagen.flow_from_directory(subset="training", **common)
    val_gen = datagen.flow_from_directory(subset="validation", **common)
    return train_gen, val_gen
