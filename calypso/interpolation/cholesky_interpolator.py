from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pickle
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
from pathlib import Path

from .uncertainty import estimate_epistemic_uncertainty


@dataclass
class CholeskyInterpolator:
    """
    Interpolator for PCA coefficient distributions using Cholesky decomposition.
    
    This approach:
    1. Decomposes covariance matrices as Σ = L L^T 
    2. Linearly interpolates Cholesky factors L(eb, qb)
    3. Reconstructs covariance as Σ_interp = L_interp L_interp^T
    4. Guarantees positive definite covariance matrices
    """
    
    # Training data
    points: np.ndarray              # (N, 2) training (eb, qb) points
    # Dictionary containing:
    # - 'means': np.ndarray, shape (N, K) - coefficient means per training point
    # - 'covariances': np.ndarray, shape (N, K, K) - coefficient covariances per point
    coefficient_stats : Dict[str, np.ndarray]
    cholesky_factors: np.ndarray    # (N, K, K) lower-triangular Cholesky factors
    
    # Interpolators
    mean_interpolators: list        # K interpolators for coefficient means
    cholesky_interpolators: list    # K*(K+1)/2 interpolators for unique Cholesky elements
    
    # Metadata
    K: int                         # Number of PCA coefficients
    epistemic_config: Dict[str, Any]  # Epistemic uncertainty configuration
    
    def predict_distribution(
        self, 
        eb: float, 
        qb: float,
        use_exact_match_fallback: Optional[bool] = None,
        use_nearest_on_invalid: Optional[bool] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict coefficient distribution at (eb, qb).
        
        Epistemic uncertainty is controlled by epistemic_config["enabled"].
        
        Returns
        -------
        mean : np.ndarray, shape (K,)
            Predicted coefficient means
        cov : np.ndarray, shape (K, K)  
            Predicted coefficient covariance matrix (positive definite)

        Runtime fallback controls
        -------------------------
        use_exact_match_fallback : bool or None
            If True, an exact (eb, qb) match to a training point bypasses interpolation
            and uses stored training stats directly. If None, defaults to True.
        use_nearest_on_invalid : bool or None
            If True, non-finite interpolated outputs (NaN/Inf) fall back to nearest
            training-point stats. If None, defaults to True.
        """
        query_point = np.array([eb, qb])
        exact_match_enabled = True if use_exact_match_fallback is None else use_exact_match_fallback
        nearest_on_invalid_enabled = True if use_nearest_on_invalid is None else use_nearest_on_invalid

        # Exact-match shortcut avoids Delaunay boundary precision issues at training points.
        exact_mask = np.all(np.isclose(self.points, query_point[None, :], atol=1e-10), axis=1)
        if exact_match_enabled and np.any(exact_mask):
            idx = int(np.flatnonzero(exact_mask)[0])
            mu = self.coefficient_stats["means"][idx].copy()
            L = self.cholesky_factors[idx]
            Sigma_aleatoric = L @ L.T
        else:
            # Interpolate means
            mu = np.zeros(self.K)
            for k in range(self.K):
                mu[k] = self.mean_interpolators[k](eb, qb)

            # Interpolate Cholesky factors
            L = np.zeros((self.K, self.K))
            idx = 0
            for i in range(self.K):
                for j in range(i + 1):  # Lower triangular only
                    L[i, j] = self.cholesky_interpolators[idx](eb, qb)
                    idx += 1

            # Reconstruct covariance matrix
            Sigma_aleatoric = L @ L.T

            # Boundary/out-of-hull fallback: use nearest training-point stats if interpolation failed.
            if nearest_on_invalid_enabled and not (
                np.all(np.isfinite(mu)) and np.all(np.isfinite(Sigma_aleatoric))
            ):
                d2 = np.sum((self.points - query_point[None, :]) ** 2, axis=1)
                nn_idx = int(np.argmin(d2))
                mu = self.coefficient_stats["means"][nn_idx].copy()
                L = self.cholesky_factors[nn_idx]
                Sigma_aleatoric = L @ L.T
        
        # Add epistemic uncertainty if enabled in config
        if self.epistemic_config.get("enabled", False):
            epistemic_var = estimate_epistemic_uncertainty(
                coefficient_stats=self.coefficient_stats,
                query_point=query_point,
                training_points=self.points,
                K=self.K,  # Pass number of coefficients
                **self.epistemic_config
            )
            # Add epistemic uncertainty to diagonal
            Sigma_total = Sigma_aleatoric + epistemic_var
        else:
            Sigma_total = Sigma_aleatoric
        
        return mu, Sigma_total
    
    def sample_coefficients(
        self, 
        eb: float, 
        qb: float, 
        n_samples: int = 1,
        rng: Optional[np.random.Generator] = None,
        use_exact_match_fallback: Optional[bool] = None,
        use_nearest_on_invalid: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Sample PCA coefficients from predicted distribution.
        
        Epistemic uncertainty is controlled by epistemic_config["enabled"].
        
        Returns
        -------
        np.ndarray, shape (n_samples, K)
            Sampled coefficient vectors

        Runtime fallback controls
        -------------------------
        use_exact_match_fallback : bool or None
            Passed through to predict_distribution().
        use_nearest_on_invalid : bool or None
            Passed through to predict_distribution().
        """
        if rng is None:
            rng = np.random.default_rng()
        
        mu, Sigma = self.predict_distribution(
            eb,
            qb,
            use_exact_match_fallback=use_exact_match_fallback,
            use_nearest_on_invalid=use_nearest_on_invalid,
        )
        
        return rng.multivariate_normal(mu, Sigma, size=n_samples)
    
    def save(self, filepath: str | Path) -> None:
        """Save interpolator to pickle file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath: str | Path) -> 'CholeskyInterpolator':
        """Load interpolator from pickle file.""" 
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def train_cholesky_interpolator(
    points: np.ndarray,
    coefficient_stats: Dict[str, np.ndarray], 
    epistemic_config: Dict[str, Any]
) -> CholeskyInterpolator:
    """
    Train Cholesky-based interpolator from coefficient statistics.
    
    Parameters
    ----------
    points : np.ndarray, shape (N, 2)
        Training (eb, qb) points
    coefficient_stats : Dict
        Dictionary containing:
        - 'means': np.ndarray, shape (N, K) - coefficient means per training point
        - 'covariances': np.ndarray, shape (N, K, K) - coefficient covariances per point
    epistemic_config : Dict
        Configuration for epistemic uncertainty estimation
        
    Returns
    -------
    CholeskyInterpolator
        Trained interpolator ready for prediction
    """
    means = coefficient_stats['means'] 
    covariances = coefficient_stats['covariances']
    
    N, K = means.shape
    
    # Compute Cholesky decomposition of all training covariance matrices
    cholesky_factors = np.zeros((N, K, K))
    for i in range(N):
        Sigma = covariances[i]
        
        # Regularize for numerical stability
        Sigma_reg = Sigma + 1e-8 * np.eye(K)
        
        try:
            L = np.linalg.cholesky(Sigma_reg)
            cholesky_factors[i] = L
        except np.linalg.LinAlgError:
            # Fallback: use eigenvalue decomposition  
            eigenvals, eigenvecs = np.linalg.eigh(Sigma_reg)
            eigenvals = np.maximum(eigenvals, 1e-8)  # Ensure positive
            L = eigenvecs @ np.diag(np.sqrt(eigenvals))
            cholesky_factors[i] = L
    
    # Train mean interpolators (one per coefficient)
    mean_interpolators = []
    for k in range(K):
        interp = LinearNDInterpolator(points, means[:, k], fill_value=np.nan)
        mean_interpolators.append(interp)
    
    # Train Cholesky factor interpolators (lower triangular elements only)
    cholesky_interpolators = []
    for i in range(K):
        for j in range(i + 1):  # Lower triangular
            values = cholesky_factors[:, i, j]
            interp = LinearNDInterpolator(points, values, fill_value=0.0)
            cholesky_interpolators.append(interp)
    
    return CholeskyInterpolator(
        points=points,
        coefficient_stats=coefficient_stats,
        cholesky_factors=cholesky_factors,
        mean_interpolators=mean_interpolators,
        cholesky_interpolators=cholesky_interpolators,
        K=K,
        epistemic_config=epistemic_config
    )
