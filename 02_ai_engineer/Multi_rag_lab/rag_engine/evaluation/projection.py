from __future__ import annotations

import numpy as np


def pca_2d(vectors: np.ndarray) -> np.ndarray:
    """Project a 2D matrix to two principal components using NumPy SVD.

    Kept in the core package so the projection math is testable without Streamlit.
    """
    if vectors.ndim != 2 or len(vectors) == 0:
        return np.empty((0, 2), dtype=float)
    centered = vectors.astype(float) - vectors.mean(axis=0, keepdims=True)
    if centered.shape[0] == 1:
        return np.zeros((1, 2), dtype=float)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    coords = u[:, :2] * s[:2]
    if coords.shape[1] == 1:
        coords = np.column_stack([coords[:, 0], np.zeros(len(coords))])
    return coords[:, :2]
