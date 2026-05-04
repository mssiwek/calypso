from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import dill

try:
    import torch
except ImportError:
    torch = None

from ..utils.config import RAW_TS_DIR, ts_filename as _ts_filename


@dataclass(frozen=True)
class TSFileSpec:
    split: str        # "train" or "test"
    component: str    # "Mb", "M1", "M2"
    nwindows: int     # e.g. 500
    norb: int         # e.g. 1 or 10
    nbins: int        # e.g. 100
    log: bool         # True if filename ends with _log

    def filename(self) -> str:
        return _ts_filename(self.split, self.component, self.nwindows,
                            self.norb, self.nbins, self.log)

    def path(self) -> Path:
        return RAW_TS_DIR / self.filename()


def _ensure_dataset_available(path: Path) -> None:
    if path.exists():
        return
    try:
        # Available in the promoted calypso package after import rewriting.
        from ..utils.zdownload import download_ts_files  # type: ignore
    except Exception:
        try:
            # Fallback if calypso is importable alongside research sources.
            from calypso.utils.zdownload import download_ts_files  # type: ignore
        except Exception:
            return
    download_ts_files([path.name])


def _load_dataset(path: Path):
    if not path.exists():
        _ensure_dataset_available(path)
    if not path.exists():
        raise FileNotFoundError(f"Cannot find dataset file: {path}")
    with open(path, "rb") as f:
        dset = dill.load(f)
    # expects attributes: dset.X (N,2) and dset.y (N,T)
    if not hasattr(dset, "X") or not hasattr(dset, "y"):
        raise ValueError(f"Dataset missing .X or .y attributes: {path}")
    return dset


def load_single_component(
    split: str,
    component: str,  
    nwindows: int,
    norb: int,
    nbins: int,
    log: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load single component dataset.
    
    Returns
    -------
    X : np.ndarray, shape (N, 2)
        Binary parameters (eb, qb)
    y : np.ndarray, shape (N, T) 
        Time series windows
    """
    spec = TSFileSpec(split, component, nwindows, norb, nbins, log)
    dset = _load_dataset(spec.path())
    
    X = np.asarray(dset.X, dtype=np.float64)
    y = np.asarray(dset.y, dtype=np.float64)
    
    return X, y


def load_multiple_components(
    split: str,
    components: List[str],
    nwindows: int, 
    norb: int,
    nbins: int,
    log: bool = True
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Load and align multiple component datasets.
    
    Parameters
    ----------
    components : List[str]
        List of component names, e.g., ["Mb", "M1", "M2"]
        
    Returns
    -------
    X : np.ndarray, shape (N, 2)
        Binary parameters (eb, qb), consistent across components
    component_data : Dict[str, np.ndarray]
        Dictionary mapping component names to time series arrays.
        Each array has shape (N, T).
    """
    if not components:
        raise ValueError("Must specify at least one component")
    
    # Load first component to establish parameter grid
    X_ref, y_first = load_single_component(split, components[0], nwindows, norb, nbins, log)
    component_data = {components[0]: y_first}
    
    # Load remaining components and verify alignment
    for component in components[1:]:
        X_comp, y_comp = load_single_component(split, component, nwindows, norb, nbins, log)
        
        # Verify parameter alignment (same (eb,qb) values in same order)
        if not np.allclose(X_ref, X_comp, atol=1e-6):
            raise ValueError(f"Parameter mismatch between {components[0]} and {component}")
            
        component_data[component] = y_comp
    
    return X_ref, component_data


def concatenate_components(component_data: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Concatenate multiple component time series along feature axis.
    
    Parameters
    ----------
    component_data : Dict[str, np.ndarray]
        Dictionary mapping component names to arrays of shape (N, T_i)
        
    Returns
    -------
    np.ndarray, shape (N, T_total)
        Concatenated time series where T_total = sum(T_i)
    """
    matrices = [component_data[name] for name in sorted(component_data.keys())]
    return np.hstack(matrices)


def _float_close(a: np.ndarray, b: float, tol: float = 5e-3) -> np.ndarray:
    # Your dataset uses formatting %.2f, so tolerate tiny float errors
    return np.abs(a - b) < tol


def load_windows_for_binary(
    eb: float,
    qb: float,
    spec: TSFileSpec,
    max_windows: Optional[int] = None,
) -> np.ndarray:
    """
    Returns X of shape (Nw, T) for the chosen (eb, qb) from the dataset file.
    """
    path = spec.path()
    dset = _load_dataset(path)

    X = dset.X
    y = dset.y

    if torch is not None and isinstance(X, torch.Tensor):
        Xn = X.detach().cpu().numpy()
    else:
        Xn = np.asarray(X)

    if torch is not None and isinstance(y, torch.Tensor):
        yn = y.detach().cpu().numpy()
    else:
        yn = np.asarray(y)

    if Xn.ndim != 2 or Xn.shape[1] != 2:
        raise ValueError(f"dset.X expected shape (N,2), got {Xn.shape}")
    if yn.ndim != 2:
        raise ValueError(f"dset.y expected shape (N,T), got {yn.shape}")

    mask = _float_close(Xn[:, 0], eb) & _float_close(Xn[:, 1], qb)
    idx = np.where(mask)[0]
    if idx.size == 0:
        raise ValueError(
            f"No windows found in {path.name} for eb={eb}, qb={qb}. "
            f"(Check rounding: dataset is keyed at ~0.01 precision.)"
        )

    if max_windows is not None:
        idx = idx[:max_windows]

    return yn[idx].astype(np.float64)
