"""
calypso - PCA-based emulator for circumbinary accretion variability
"""

from .pca.global_pca import PCAModel, fit_pca_svd_multicomponent, reconstruct_components
from .interpolation import CholeskyInterpolator, train_cholesky_interpolator
from .reconstruction import PCAEmulator
from .runtime import load_emulator

__version__ = "1.0.0"

__all__ = [
    "PCAModel",
    "fit_pca_svd_multicomponent",
    "reconstruct_components",
    "CholeskyInterpolator",
    "train_cholesky_interpolator",
    "PCAEmulator",
    "load_emulator",
]
