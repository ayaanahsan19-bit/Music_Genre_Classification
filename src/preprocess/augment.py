"""
augment.py — Audio data augmentation utilities.

Provides time-stretch, pitch-shift, and noise injection transforms
for on-the-fly or offline augmentation of .wav files.
"""

import numpy as np
import librosa


def time_stretch(y: np.ndarray, rate: float = 1.2) -> np.ndarray:
    """Time-stretch the audio signal by *rate* (>1 = faster, <1 = slower)."""
    return librosa.effects.time_stretch(y, rate=rate)


def pitch_shift(y: np.ndarray, sr: int = 22050, n_steps: float = 2.0) -> np.ndarray:
    """Shift pitch by *n_steps* semitones (positive = up, negative = down)."""
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)


def add_noise(y: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
    """Inject Gaussian white noise."""
    noise = np.random.randn(len(y)) * noise_factor
    return y + noise


def random_gain(y: np.ndarray, low: float = 0.7, high: float = 1.3) -> np.ndarray:
    """Randomly scale amplitude."""
    gain = np.random.uniform(low, high)
    return y * gain


def augment_audio(y: np.ndarray, sr: int = 22050,
                  stretch_rate: float | None = None,
                  pitch_steps: float | None = None,
                  noise_factor: float | None = None) -> np.ndarray:
    """Apply a chain of augmentations (only those whose parameter is not None)."""
    if stretch_rate is not None:
        y = time_stretch(y, rate=stretch_rate)
    if pitch_steps is not None:
        y = pitch_shift(y, sr=sr, n_steps=pitch_steps)
    if noise_factor is not None:
        y = add_noise(y, noise_factor=noise_factor)
    return y
