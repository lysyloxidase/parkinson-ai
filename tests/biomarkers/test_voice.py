"""Voice biomarker tests."""

from __future__ import annotations

import numpy as np

from parkinson_ai.biomarkers.digital.voice import extract_voice_features


def test_extract_voice_features() -> None:
    """Voice extraction should return key acoustic features."""

    signal = np.sin(np.linspace(0, 4 * np.pi, 400))
    features = extract_voice_features(signal, sample_rate=200)
    assert {"jitter", "shimmer", "hnr", "mfcc_0"}.issubset(features)
