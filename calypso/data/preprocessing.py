from __future__ import annotations
from typing import Tuple
import numpy as np


def preprocess_matrix(Y: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Y: (Nw, T) windows from dataset.
    Returns (Yproc, metadata).
    
    Note: No per-window centering or normalization is applied.
    PCA will handle global centering during fitting.
    """
    if Y.ndim != 2:
        raise ValueError(f"Expected Y shape (Nw, T), got {Y.shape}")

    X = Y.astype(np.float64).copy()

    meta = {
        "shape": [int(X.shape[0]), int(X.shape[1])],
    }
    return X, meta