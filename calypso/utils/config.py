from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from .constants import DEFAULT_DATA_DIR

RAW_TS_DIR = Path(DEFAULT_DATA_DIR)


def ts_filename(
    split: str,
    comp: str,
    nwindows: int,
    norb: int = 10,
    nbins: int = 100,
    log: bool = True,
) -> str:
    base = f"TS_{split}_{comp}_nwindows{nwindows}_norb{norb}_nbinsPb{nbins}"
    if log:
        base += "_log"
    return base + ".pkl"


def ts_path(
    split: str,
    comp: str,
    nwindows: int,
    norb: int = 10,
    nbins: int = 100,
    log: bool = True,
    ts_dir: Path | str | None = None,
) -> Path:
    data_dir = Path(ts_dir) if ts_dir is not None else RAW_TS_DIR
    return data_dir / ts_filename(split, comp, nwindows, norb, nbins, log)


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    required_sections = ["dataset", "pca"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

    if "components" not in config["pca"]:
        raise ValueError("Missing required config: pca.components")

    valid_components = {"Mb", "M1", "M2"}
    components = config["pca"]["components"]
    if not isinstance(components, list) or not components:
        raise ValueError("pca.components must be a non-empty list")

    for comp in components:
        if comp not in valid_components:
            raise ValueError(f"Invalid component: {comp}. Must be one of {valid_components}")

    return config


def get_default_config() -> Dict[str, Any]:
    return {
        "components": ["Mb"],
        "dataset": {
            "split_train": "train",
            "split_val": "test",
            "nwindows": 100,
            "norb": 10,
            "nbins": 100,
            "log": True,
        },
        "pca": {
            "k": 100,
        },
        "interpolation": {
            "method": "cholesky_linear",
            "epistemic_uncertainty": {
                "enabled": True,
                "distance_weight": 1.0,
                "min_uncertainty": 0.01,
            },
        },
        "validation": {
            "split": "test",
            "ncases": 12,
            "case_seed": 0,
            "nsynth": 500,
            "eb_values": [0.55, 0.35, 0.15],
            "qb_values": [0.95, 0.75, 0.45],
        },
        "plotting": {
            "style": "seaborn-v0_8-whitegrid",
            "figure_size": [12, 8],
            "dpi": 150,
            "colors": {
                "true": "#1f77b4",
                "predicted": "#ff7f0e",
                "uncertainty": "#d62728",
            },
            "fonts": {
                "title_size": 14,
                "label_size": 12,
                "tick_size": 10,
            },
            "layout": "grid",
        },
    }
