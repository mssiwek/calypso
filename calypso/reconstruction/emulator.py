from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import pickle

from ..pca.global_pca import PCAModel, reconstruct_components
from ..interpolation import CholeskyInterpolator


class PCAEmulator:
    """
    Complete PCA-based emulator for multi-component accretion time series.
    
    This class combines:
    - PCA basis for dimensionality reduction
    - Cholesky interpolation for coefficient prediction
    - Multi-component reconstruction with uncertainty quantification
    """
    
    def __init__(
        self, 
        pca_model: PCAModel, 
        interpolator: CholeskyInterpolator,
        config: Dict[str, Any]
    ):
        self.pca_model = pca_model
        self.interpolator = interpolator
        self.config = config
        
        # Validate compatibility
        if interpolator.K != min(pca_model.components_.shape[0], pca_model.components_.shape[1]):
            raise ValueError("PCA model and interpolator have incompatible dimensions")
    
    def predict(
        self, 
        eb: float, 
        qb: float, 
        n_samples: int = 1,
        return_coefficients: bool = False,
        rng: Optional[np.random.Generator] = None,
        use_exact_match_fallback: Optional[bool] = None,
        use_nearest_on_invalid: Optional[bool] = None,
        interpolator_config: Optional[Dict[str, Any]] = None,
        epistemic_enabled: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Predict accretion time series for given binary parameters.
        
        Epistemic uncertainty is controlled by the interpolator's config.
        
        Parameters
        ----------
        eb, qb : float
            Binary eccentricity and mass ratio
        n_samples : int
            Number of stochastic samples to generate
        return_coefficients : bool  
            Whether to return PCA coefficients in output
        rng : np.random.Generator, optional
            Random number generator for reproducible sampling
        use_exact_match_fallback : bool or None
            Runtime control for exact training-point fallback in interpolation.
            If None, interpolator defaults are used.
        use_nearest_on_invalid : bool or None
            Runtime control for nearest-training-point fallback when interpolation
            returns NaN/Inf. If None, interpolator defaults are used.
        interpolator_config : dict, optional
            Runtime overrides merged into interpolator.epistemic_config for this call only.
        epistemic_enabled : bool, optional
            Convenience runtime switch overriding epistemic_config["enabled"] for this call only.
            
        Returns
        -------
        Dict containing:
            - For each component: predicted time series array (n_samples, T_i)
            - 'coefficients': PCA coefficients if requested (n_samples, K)
            - 'parameters': input parameters {'eb': eb, 'qb': qb}
        """
        if rng is None:
            rng = np.random.default_rng()
        
        # Sample PCA coefficients (optionally override epistemic config at runtime).
        old_config = dict(self.interpolator.epistemic_config or {})
        runtime_config = dict(old_config)
        if interpolator_config:
            runtime_config.update(interpolator_config)
        if epistemic_enabled is not None:
            runtime_config["enabled"] = bool(epistemic_enabled)
        self.interpolator.epistemic_config = runtime_config
        try:
            coefficients = self.interpolator.sample_coefficients(
                eb,
                qb,
                n_samples,
                rng,
                use_exact_match_fallback=use_exact_match_fallback,
                use_nearest_on_invalid=use_nearest_on_invalid,
            )
        finally:
            self.interpolator.epistemic_config = old_config
        
        # Reconstruct time series for all components with constraint enforcement
        enforce_constraint = self.config.get('enforce_constraint', True)
        reconstructions = reconstruct_components(coefficients, self.pca_model, enforce_constraint)
        
        result = {
            'parameters': {'eb': eb, 'qb': qb},
        }
        
        # Add component reconstructions
        for component_name, ts_samples in reconstructions.items():
            result[component_name] = ts_samples
        
        # Add coefficients if requested
        if return_coefficients:
            result['coefficients'] = coefficients
        
        return result
    
    def predict_mean(
        self,
        eb: float,
        qb: float,
        use_exact_match_fallback: Optional[bool] = None,
        use_nearest_on_invalid: Optional[bool] = None,
        interpolator_config: Optional[Dict[str, Any]] = None,
        epistemic_enabled: Optional[bool] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Predict mean time series without sampling (faster for evaluation).
        
        Returns
        -------
        Dict[str, np.ndarray]
            Mean predicted time series for each component

        Parameters
        ----------
        use_exact_match_fallback : bool or None
            Runtime control for exact training-point fallback in interpolation.
        use_nearest_on_invalid : bool or None
            Runtime control for nearest-training-point fallback when interpolation
            returns NaN/Inf.
        interpolator_config : dict, optional
            Runtime overrides merged into interpolator.epistemic_config for this call only.
        epistemic_enabled : bool, optional
            Convenience runtime switch overriding epistemic_config["enabled"] for this call only.
        """
        old_config = dict(self.interpolator.epistemic_config or {})
        runtime_config = dict(old_config)
        if interpolator_config:
            runtime_config.update(interpolator_config)
        if epistemic_enabled is not None:
            runtime_config["enabled"] = bool(epistemic_enabled)
        self.interpolator.epistemic_config = runtime_config
        try:
            mu_coeffs, _ = self.interpolator.predict_distribution(
                eb,
                qb,
                use_exact_match_fallback=use_exact_match_fallback,
                use_nearest_on_invalid=use_nearest_on_invalid,
            )
        finally:
            self.interpolator.epistemic_config = old_config
        
        # Reconstruct using mean only with constraint enforcement
        enforce_constraint = self.config.get('enforce_constraint', True)
        reconstructions = reconstruct_components(mu_coeffs.reshape(1, -1), self.pca_model, enforce_constraint)
        
        # Remove batch dimension
        return {name: ts[0] for name, ts in reconstructions.items()}
    
    def save(self, filepath: str | Path) -> None:
        """Save complete emulator to file.""" 
        save_data = {
            'pca_model': self.pca_model.to_dict(),
            'interpolator': self.interpolator,
            'config': self.config,
            'emulator_version': '1.0'
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
    
    @classmethod
    def load(cls, filepath: str | Path) -> 'PCAEmulator':
        """Load complete emulator from file."""
        class _RemappingUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == "calypsopca" or module.startswith("calypsopca."):
                    module = "calypso" + module[len("calypsopca"):]
                return super().find_class(module, name)

        with open(filepath, 'rb') as f:
            save_data = _RemappingUnpickler(f).load()
        
        # Reconstruct PCA model
        pca_dict = save_data['pca_model']
        pca_model = PCAModel(**pca_dict)
        
        return cls(
            pca_model=pca_model,
            interpolator=save_data['interpolator'],
            config=save_data['config']
        )
    
    @property
    def component_names(self) -> list[str]:
        """Get list of component names."""
        return self.pca_model.component_names_
    
    @property 
    def n_components(self) -> int:
        """Get number of components."""
        return self.pca_model.n_components_total
