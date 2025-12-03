from __future__ import annotations
from typing import Dict
import numpy as np
from .io import load_model_any
from .decoding import decode_prediction
from ..utils.plotting import make_time_grid

class Emulator:
    def __init__(self, cfg: dict):
        self.cfg = dict(cfg)  # shallow copy

    def load_model(self, path: str) -> None:
        print(f"Loading model from {path}")
        self.model, self.meta = load_model_any(path)

    def predict(self, eb: float, qb: float) -> Dict[str, np.ndarray]:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call _load_component_model() first.")

        pred = self.model.predict(eb, qb)  # expects (T,2): [mu, var]
        time = make_time_grid(self.cfg["norb"], self.cfg["nbins"])
        mean, p16, p84 = decode_prediction(pred, log_model=bool(self.cfg.get("log_model", False)))
        return {"time": time, "mean": mean, "p16": p16, "p84": p84}