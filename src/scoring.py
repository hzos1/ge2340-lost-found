"""Score formulas shared by training and the web app.

CLIP fusion uses fixed weights on similarities, as in the proposal.
The linear model uses the four pair features and a learned weight vector.
"""

from __future__ import annotations

import numpy as np


def minmax(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    low = float(array.min())
    high = float(array.max())
    if high - low < 1e-8:
        return np.zeros_like(array)
    return (array - low) / (high - low)


def clip_fusion(
    visual: np.ndarray,
    location_closeness: np.ndarray,
    time_closeness: np.ndarray,
    weights: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return the fused score and the per-query normalised visual similarity."""
    visual_norm = minmax(visual)
    score = (
        weights[0] * visual_norm
        + weights[1] * np.asarray(location_closeness, dtype=np.float32)
        + weights[2] * np.asarray(time_closeness, dtype=np.float32)
    )
    return score.astype(np.float32), visual_norm


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -30.0, 30.0)
    return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32)


def linear_probability(features: np.ndarray, coefficients: np.ndarray, bias: float) -> np.ndarray:
    logits = bias + np.asarray(features, dtype=np.float32) @ np.asarray(coefficients, dtype=np.float32)
    return sigmoid(logits)
