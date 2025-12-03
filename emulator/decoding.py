from __future__ import annotations
import numpy as np, torch
from typing import Tuple

def decode_prediction(pred, log_model: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    pred: shape (T,2): [:,0]=mu, [:,1]=var (log10-space if log_model)
    Returns (mean, p16, p84) in linear space.
    """
    if not isinstance(pred, torch.Tensor):
        pred = torch.as_tensor(pred)

    mu = pred[:, 0]
    v  = torch.clamp(pred[:, 1], min=1e-6 if log_model else 0.0)
    sd = torch.sqrt(v)

    if log_model:
        mean = (10.0 ** mu).cpu().numpy()
        p16  = (10.0 ** (mu - sd)).cpu().numpy()
        p84  = (10.0 ** (mu + sd)).cpu().numpy()
    else:
        mean = mu.cpu().numpy()
        p16  = (mu - sd).cpu().numpy()
        p84  = (mu + sd).cpu().numpy()

    return mean, p16, p84