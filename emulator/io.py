from __future__ import annotations
import os, torch, glob
from typing import Any, Tuple

import os
import torch
from typing import Any

_CUDA_ERR = "Attempting to deserialize object on a CUDA device"

def _cpu_mapper():
    # Robust CPU remapper for legacy pickles that try to restore CUDA storages
    return lambda storage, loc: "cpu"

def cpu_safe_load(fp: str, *, allow_full_pickle: bool | None = None) -> Any:
    """
    Robust CPU-only load for PyTorch 2.6+ and legacy CUDA pickles.

    Order:
      1) weights_only=True with simple 'cpu' mapping
      2) weights_only=True with lambda map_location (handles nested loads)
      3) If allowed, full pickle (weights_only=False) with 'cpu'
      4) If CUDA restore still shows up, full pickle + lambda map_location
    """
    if allow_full_pickle is None:
        allow_full_pickle = os.environ.get("CALYPSO_ALLOW_PICKLE", "0") == "1"

    # 1) Safe path: state_dict-style
    try:
        return torch.load(fp, map_location="cpu", weights_only=True)
    except Exception as e1:
        # 2) Retry with lambda mapper (covers nested legacy loads)
        try:
            return torch.load(fp, map_location=_cpu_mapper(), weights_only=True)
        except Exception as e2:
            if not allow_full_pickle:
                raise RuntimeError(
                    "Failed to load with weights_only=True. If this is a trusted full-pickle "
                    "checkpoint, set CALYPSO_ALLOW_PICKLE=1 to permit weights_only=False."
                ) from e2

            # 3) Trusted: full pickle, regular CPU mapping
            try:
                return torch.load(fp, map_location="cpu", weights_only=False)
            except Exception as e3:
                # 4) Last resort: full pickle + lambda mapper
                try:
                    return torch.load(fp, map_location=_cpu_mapper(), weights_only=False)
                except Exception as e4:
                    # Surface the most informative CUDA hint if present
                    if isinstance(e4, RuntimeError) and _CUDA_ERR in str(e4):
                        raise RuntimeError(
                            "CUDA storages detected in checkpoint and could not be remapped. "
                            "Ensure the file is not a zip of nested pickles saved on GPU. "
                            "As a one-time fix, open this checkpoint on a GPU-enabled machine "
                            "and re-save as {'state_dict': ..., 'meta': ...}."
                        ) from e4
                    raise

def resolve_checkpoint_path(model_dir: str, model_name: str, epoch: int | None = None, best_model: bool = False) -> str:
    # Recursively search for model name in model_dir and its subdirectories
    for f in glob.glob(os.path.join(model_dir, "**", f"{model_name}"), recursive=True):
        if os.path.isfile(f):
            print(f"Resolved checkpoint for {model_name} to {f}")
            return f

    raise FileNotFoundError(f"Could not resolve checkpoint for {model_name} in {model_dir}")

from typing import Any, Dict, Tuple, Optional

def load_model_any(fp: str, *, model_class: Optional[str] = None) -> Tuple[Any, Dict]:
    obj = cpu_safe_load(fp, allow_full_pickle=True)  # or via env CALYPSO_ALLOW_PICKLE=1
    
    print(f"Loaded object type from {fp}: {type(obj)}")
    exit()

    if isinstance(obj, tuple) and len(obj) == 2:
        model, meta = obj
        return model, (meta if isinstance(meta, dict) else {})

    if isinstance(obj, dict):
        if "model" in obj:
            return obj["model"], obj.get("meta", {})
        if "state_dict" in obj:
            return obj["state_dict"], obj.get("meta", {})
        try:
            if all(hasattr(v, "shape") for v in obj.values()):
                return obj, {}
        except Exception:
            pass

    return obj, {}