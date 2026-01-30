from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union
import torch
from ..utils.constants import TMOD_PATH, norb, nbins
from .decoding import decode_prediction
from ..utils.plotting import make_time_grid
from .io import cpu_safe_load
import importlib.util
import sys
from functools import reduce
from ..utils.constants import DEFAULT_WEIGHTS_DIR

@dataclass
class Emulator:
    model: torch.nn.Module
    device: torch.device
    sequence_length: Optional[int] = None
    meta: Dict[str, Any] = None

	# "@torch.inference_mode()" is equivalent to "with torch.no_grad():"
    @torch.inference_mode()
    def predict(self, eb: float, qb: float, sequence_length: Optional[int] = None):
        self.model.eval()
        T = sequence_length or self.sequence_length
        if T is None:
            raise ValueError("sequence_length must be provided.")
        x = torch.tensor([[eb, qb]], dtype=torch.float32, device=self.device)
        y, _ = self.model(x, sequence_length=T, hidden=None)
        y = y.squeeze(0).cpu()
        
        time = make_time_grid(norb, nbins*norb)
        mean, p16, p84 = decode_prediction(y, log_model=True)
        
        return {"time": time, "mean": mean, "p16": p16, "p84": p84}

def _getattr_dotted(obj, dotted: str):
    # supports nested qualnames like "Outer.Inner"
    return reduce(getattr, dotted.split("."), obj)

def _load_module_from_file(py_path: Path, module_name: str = "_calypso_models_from_file"):
    """
    Import a Python module *from an explicit file path*, without relying on package imports.
    """
    py_path = Path(py_path).expanduser().resolve()
    if not py_path.exists():
        raise FileNotFoundError(f"TMOD_PATH does not exist: {py_path}")

    spec = importlib.util.spec_from_file_location(module_name, str(py_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {py_path}")

    mod = importlib.util.module_from_spec(spec)
    # cache so repeated loads don't re-exec the file (and helps intra-module references)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

def _extract_attr_tail(ckpt_fqcn: str) -> str:
    """
    From legacy checkpoint strings like:
        '.n.home00....calypso.models.models.MyModel'
        'models.models.MyModel'
        'whatever.MyModel'
    return just the attribute tail we want to fetch from models.py:
        'MyModel'
    """
    s = (ckpt_fqcn or "").strip().lstrip(".")
    _, _, tail = s.rpartition(".")
    if not tail:
        raise ValueError(f"Bad ckpt model_class_name (no tail): {ckpt_fqcn!r}")
    # If someone ever stored a path ending in ".py", tail would be "py" — catch that loudly.
    if tail == "py":
        raise ValueError(f"ckpt model_class_name looks like a file path, not a class: {ckpt_fqcn!r}")
    return tail

def _import_fqcn(ckpt_fqcn: str):
    print("ckpt_fqcn =", ckpt_fqcn)

    # Always load canonical definitions from TMOD_PATH
    # IMPORTANT: TMOD_PATH should be a real filesystem path to models/models.py
    models_mod = _load_module_from_file(Path(TMOD_PATH), module_name="_calypso_models")

    attr = _extract_attr_tail(ckpt_fqcn)
    print("using TMOD_PATH =", str(Path(TMOD_PATH).resolve()))
    print("attr =", attr)

    return _getattr_dotted(models_mod, attr)


def load_emulator(component: str, device: str = "auto") -> Emulator:
    fp = Path(DEFAULT_WEIGHTS_DIR) / f"calypso_{component}.pkl"
    ckpt = cpu_safe_load(str(fp), allow_full_pickle=True)

    # Pick a device for inference
    dev = torch.device("cuda:0" if (device == "auto" and torch.cuda.is_available()) else "cpu") \
          if device == "auto" else torch.device(device)

    # ✅ Your saved format: dict with model_class_name/model_kwargs/model_state_dict
    if isinstance(ckpt, dict) and "model_class_name" in ckpt and "model_state_dict" in ckpt:
        ModelClass = _import_fqcn(ckpt["model_class_name"])
        model_kwargs = ckpt.get("model_kwargs", {}) or {}
        model = ModelClass(**model_kwargs)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(dev)

        meta = {
            "epoch": ckpt.get("epoch"),
            "saved_as_best": ckpt.get("saved_as_best"),
            "train_config": ckpt.get("train_config"),
        }
        return Emulator(model=model, device=dev, sequence_length=ckpt.get("sequence_length"), meta=meta)

    # If it's already a torch.nn.Module, wrap it
    if isinstance(ckpt, torch.nn.Module):
        ckpt.to(dev)
        return Emulator(model=ckpt, device=dev, sequence_length=None, meta={})

    # If it's a raw state_dict, you *cannot* predict without knowing model class/kwargs
    if isinstance(ckpt, dict) and all(hasattr(v, "shape") for v in ckpt.values()):
        raise RuntimeError("Loaded a raw state_dict but no model class/kwargs info. Cannot build emulator.")

    raise RuntimeError(f"Unrecognized checkpoint contents: {type(ckpt)}")