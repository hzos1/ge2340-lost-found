"""Score formulas shared by training and the web app.

CLIP fusion uses fixed weights on similarities, as in the proposal.
The linear model uses the four pair features and a learned weight vector.
"""

from __future__ import annotations

import numpy as np

HYBRID_CLIP_WEIGHT = 0.5
HYBRID_CNN_WEIGHT = 0.5


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


def hybrid_match_score(clip_scores: np.ndarray, cnn_scores: np.ndarray) -> np.ndarray:
    """Combine the CLIP and CNN matcher scores with equal weight."""
    clip_values = np.asarray(clip_scores, dtype=np.float32)
    cnn_values = np.asarray(cnn_scores, dtype=np.float32)
    return (HYBRID_CLIP_WEIGHT * clip_values + HYBRID_CNN_WEIGHT * cnn_values).astype(np.float32)
