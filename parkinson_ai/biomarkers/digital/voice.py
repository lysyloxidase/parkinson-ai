"""Voice biomarker feature extraction."""

from __future__ import annotations

from collections.abc import Sequence
from math import log10
from typing import Any

import numpy as np
from numpy.typing import NDArray


def extract_voice_features(
    signal: Sequence[float] | NDArray[np.floating[Any]],
    sample_rate: int,
) -> dict[str, float]:
    """Extract lightweight voice features without external DSP dependencies."""

    array = np.asarray(signal, dtype=float)
    if array.size == 0:
        raise ValueError("signal must not be empty")
    magnitude = np.abs(array)
    diffs = np.abs(np.diff(array)) if array.size > 1 else np.array([0.0])
    jitter = float(np.mean(np.abs(np.diff(diffs)))) if diffs.size > 1 else 0.0
    shimmer = float(np.std(magnitude) / (np.mean(magnitude) + 1e-8))
    noise = array - np.mean(array)
    signal_power = float(np.mean(array**2))
    noise_power = float(np.var(noise)) + 1e-8
    hnr = 10.0 * log10((signal_power + 1e-8) / noise_power)
    zero_crossings = np.where(np.diff(np.signbit(array)))[0]
    pitch_hz = float((len(zero_crossings) / max(len(array), 1)) * sample_rate / 2.0)
    energy = float(np.mean(array**2))
    return {
        "jitter": jitter,
        "shimmer": shimmer,
        "hnr": hnr,
        "pitch_hz": pitch_hz,
        "mfcc_0": float(np.log(energy + 1e-8)),
        "signal_energy": energy,
    }
