from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class PCAModel:
    mean_: np.ndarray                 # (T_total,) where T_total = sum(T_i) for all components
    components_: np.ndarray           # (K, T_total)  rows are principal directions
    singular_values_: np.ndarray      # (K,)
    explained_variance_: np.ndarray   # (K,)
    explained_variance_ratio_: np.ndarray  # (K,)
    n_samples_: int
    n_features_: int
    # Multi-component tracking
    component_names_: list[str]       # e.g., ["Mb", "M1", "M2"]
    component_lengths_: list[int]     # e.g., [1000, 1000, 1000] for T=1000 each
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_": self.mean_,
            "components_": self.components_,
            "singular_values_": self.singular_values_,
            "explained_variance_": self.explained_variance_,
            "explained_variance_ratio_": self.explained_variance_ratio_,
            "n_samples_": self.n_samples_,
            "n_features_": self.n_features_,
            "component_names_": self.component_names_,
            "component_lengths_": self.component_lengths_,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PCAModel":
        return cls(
            mean_=d["mean_"],
            components_=d["components_"],
            singular_values_=d["singular_values_"],
            explained_variance_=d["explained_variance_"],
            explained_variance_ratio_=d["explained_variance_ratio_"],
            n_samples_=d["n_samples_"],
            n_features_=d["n_features_"],
            component_names_=d["component_names_"],
            component_lengths_=d["component_lengths_"],
        )
    
    @property
    def n_components_total(self) -> int:
        """Total number of components (e.g., 3 for Mb+M1+M2)."""
        return len(self.component_names_)
    
    @property 
    def component_boundaries_(self) -> list[tuple[int, int]]:
        """Get (start, end) indices for each component in concatenated array."""
        boundaries = []
        start = 0
        for length in self.component_lengths_:
            end = start + length
            boundaries.append((start, end))
            start = end
        return boundaries


def fit_pca_svd(X: np.ndarray, k: Optional[int] = None) -> PCAModel:
    """
    Fit PCA using SVD on centered data.

    X: (N, T)
    k: number of components to keep. If None, keeps full rank (min(N,T)).

    Returns PCAModel with components_ shaped (K, T).
    """
    if X.ndim != 2:
        raise ValueError(f"Expected X shape (N,T), got {X.shape}")

    N, T = X.shape
    mean = X.mean(axis=0)
    Xc = X - mean

    # Full SVD; for huge N you can swap in randomized later if needed.
    # Xc = U S Vt
    # U are the left singular vectors (N, r)
    # S is a matrix of the singular values (r,)
    # Vt are the right singular vectors (r, T)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)

    r = min(N, T)
    if k is None:
        k = r
    k = int(min(k, r))
    Vt_k = Vt[:k]
    S_k = S[:k]

    # Explained variance for PCA from SVD:
    # eigenvalues of covariance = S^2 / (N-1)
    denom = max(N - 1, 1)
    ev = (S_k**2) / denom

    total_ev = (S**2).sum() / denom
    evr = ev / total_ev if total_ev > 0 else np.zeros_like(ev)

    return PCAModel(
        mean_=mean.astype(np.float64),
        components_=Vt_k.astype(np.float64),
        singular_values_=S_k.astype(np.float64),
        explained_variance_=ev.astype(np.float64),
        explained_variance_ratio_=evr.astype(np.float64),
        n_samples_=int(N),
        n_features_=int(T),
        component_names_=["Mb"],  # Default for backwards compatibility
        component_lengths_=[T],   # Single component uses full feature length
    )


def project(X: np.ndarray, model: PCAModel, k: Optional[int] = None) -> np.ndarray:
    """
    
    X: (N, T) input data to project
    model: fitted PCA model with components_ (K, T)
    k: number of components to project onto. If None, uses all available in model.
    
    Project windows X onto first k components. Returns coefficients C (N, k).
    """
    if X.ndim != 2:
        raise ValueError(f"Expected X shape (N,T), got {X.shape}")
    Xc = X - model.mean_[None, :]
    comps = model.components_ if k is None else model.components_[:k]
    # matrix multiplication, projecting onto each component direction
    # Xc has shape (N, T), comps.T has shape (T, k), result is (N, k)
    # C is called the "scores" or "coefficients" of the PCA projection
    C = Xc @ comps.T
    return C


def reconstruct(C: np.ndarray, model: PCAModel) -> np.ndarray:
    """
    Reconstruct windows from coefficients C (N,k). Returns Xhat (N,T_total).
    """
    Xhat = model.mean_[None, :] + C @ model.components_[: C.shape[1], :]
    return Xhat


def reconstruct_components(
    C: np.ndarray, 
    model: PCAModel, 
    enforce_constraint: bool = True
) -> Dict[str, np.ndarray]:
    """
    Reconstruct and split into individual components.
    
    Parameters
    ----------
    C : np.ndarray, shape (N, k)
        PCA coefficients
    model : PCAModel 
        Fitted PCA model with component tracking
    enforce_constraint : bool, default True
        Whether to enforce physical constraint Ṁ1 + Ṁ2 = Ṁb for multi-component models
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping component names to reconstructed time series.
        Each value has shape (N, T_i) where T_i is length of component i.
    """
    # Full reconstruction
    Xhat_full = reconstruct(C, model)  # (N, T_total)
    
    # Split by component boundaries
    result = {}
    for name, (start, end) in zip(model.component_names_, model.component_boundaries_):
        result[name] = Xhat_full[:, start:end]
    
    return result


def fit_pca_svd_multicomponent(
    component_data: Dict[str, np.ndarray], 
    k: Optional[int] = None
) -> PCAModel:
    """
    Fit PCA on concatenated multi-component data.
    
    Parameters
    ---------- 
    component_data : Dict[str, np.ndarray]
        Dictionary mapping component names to data matrices.
        Each matrix has shape (N, T_i) where N is consistent across components.
    k : int, optional
        Number of components to keep.
        
    Returns
    -------
    PCAModel
        Fitted model with component tracking.
    """
    # Validate input
    component_names = list(component_data.keys())
    matrices = list(component_data.values())
    
    if not matrices:
        raise ValueError("No component data provided")
    
    # Check consistent sample counts
    N = matrices[0].shape[0]
    for name, mat in component_data.items():
        if mat.shape[0] != N:
            raise ValueError(f"Inconsistent sample count for {name}: {mat.shape[0]} vs {N}")
    
    # Concatenate along feature axis
    X_concat = np.hstack(matrices)  # (N, T_total)
    component_lengths = [mat.shape[1] for mat in matrices]
    
    # Fit PCA on concatenated data
    base_model = fit_pca_svd(X_concat, k)
    
    # Enhanced model with component tracking
    return PCAModel(
        mean_=base_model.mean_,
        components_=base_model.components_,
        singular_values_=base_model.singular_values_,
        explained_variance_=base_model.explained_variance_, 
        explained_variance_ratio_=base_model.explained_variance_ratio_,
        n_samples_=base_model.n_samples_,
        n_features_=base_model.n_features_,
        component_names_=component_names,
        component_lengths_=component_lengths,
    )