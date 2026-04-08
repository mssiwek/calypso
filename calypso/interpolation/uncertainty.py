from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from typing import Dict, Any


def estimate_epistemic_uncertainty(
    coefficient_stats: Dict[str, np.ndarray],
    query_point: np.ndarray,
    training_points: np.ndarray,
    K: int,  # Number of coefficients
    epsilon: float = 1e-8,
    alpha: float = 2.0,
    
    **kwargs
) -> np.ndarray:
    """
    Estimate epistemic (model) uncertainty at a query point in (eb, qb) space.

    Measures the spatial scatter of the first moment (mean PCA coefficient
    vectors) across training binaries, weighted by inverse distance to the
    query point.  Returns a full K×K covariance matrix capturing correlated
    uncertainty across PCA modes.

    Only the per-binary means are used; per-binary covariance matrices
    (present in coefficient_stats['covariances']) are not incorporated.

    Parameters
    ----------
    coefficient_stats : Dict[str, np.ndarray]
        Dictionary containing:
        - 'means': np.ndarray, shape (N, K) - coefficient means per training point
        - 'covariances': np.ndarray, shape (N, K, K) - unused by this function
    query_point : np.ndarray, shape (2,)
        Query point (eb, qb)
    training_points : np.ndarray, shape (N, 2)
        Training data points
    K : int
        Number of PCA coefficients
    epsilon : float
        Small value to avoid division by zero in distance weighting
    alpha : float
        Exponent controlling how quickly weights decay with distance

    Returns
    -------
    np.ndarray, shape (K, K)
        Epistemic covariance matrix (weighted covariance of training-point
        mean coefficient vectors around the distance-weighted mean)
    """
    # Compute distances to all training points
    distances = cdist(query_point.reshape(1, -1), training_points, metric='euclidean') # (1, N)
    distances = distances.squeeze()  # (N,)
    
    # weights based on distance 
    weights = 1/(distances + epsilon)**alpha  # shape (N,)
    
    # normalized weights
    W = np.sum(weights)
    nweights = weights / W # shape (N,)
    
    
    def neighbor_mean_covariance(coefficient_stats, nweights):
        # Extract covariances from coefficient_stats
        means = coefficient_stats['means']  # (N, K)
        
        # Compute weighted mean of neighbor means
        means_nw = np.sum(nweights[:, None] * means, axis=0)  # (K,)
        
        # deviations from weighted mean of the neighbor mean pca coefficients
        deviations = means - means_nw # (N, K)
        
        # Compute weighted covariance of neighbor means
        Sigma_epistemic = (deviations.T * nweights) @ deviations # (K, K)

        return Sigma_epistemic
    
    # Return epistemic uncertainty via distance-weighted variation of mean pca components among neighbors
    return neighbor_mean_covariance(coefficient_stats, nweights)