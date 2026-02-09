# calypso/emulator/data.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import dill
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from ..utils.constants import DEFAULT_DATA_DIR as DATA_DIR

Comp = Literal["Mb", "M1", "M2"]
Source = Literal["train", "test", "both"]


def _round_key(eb: float, qb: float, ndigits: int = 2) -> Tuple[float, float]:
    return (round(float(eb), ndigits), round(float(qb), ndigits))


def _pkl_path(split: Literal["train", "test"], comp: Comp, nwindows: int, norb: int, nbins: int, log: bool) -> Path:
    name = f"TS_{split}_{comp}_nwindows{nwindows}_norb{norb}_nbinsPb{nbins}"
    name += "_log.pkl" if log else ".pkl"
    return DATA_DIR / name


def _load_split_dict(
    split: Literal["train", "test"],
    comp: Comp,
    nwindows: int,
    norb: int,
    nbins: int,
    log: bool,
    round_ndigits: int,
) -> Dict[Tuple[float, float], np.ndarray]:
    fp = _pkl_path(split, comp, nwindows, norb, nbins, log)
    if not fp.exists():
        raise FileNotFoundError(f"Dataset not found: {fp}")

    with fp.open("rb") as f:
        obj = dill.load(f)

    out: Dict[Tuple[float, float], np.ndarray] = {}
    for i in range(len(obj)):
        eb, qb = obj.X[i].tolist()
        key = _round_key(eb, qb, round_ndigits)

        y = obj.y[i].detach().cpu().numpy() if torch.is_tensor(obj.y[i]) else np.asarray(obj.y[i])
        out[key] = (10 ** y) if log else y

    return out


@dataclass(frozen=True)
class SeriesWithMeta:
    """Return type for .get(): the series plus minimal provenance."""
    series: np.ndarray
    source: Source          # "train" | "test" | "both"
    chosen: Literal["train", "test"]  # which split actually supplied `series` if both exist


class TimeSeriesDB:
    """
    Unified (train + test) time-series database.

    - No split awareness required.
    - get(eb,qb) returns SeriesWithMeta containing provenance.
    - DataLoader is over the merged dataset only.
    """

    def __init__(
        self,
        comp: Comp,
        nwindows: int,
        norb: int,
        nbins: int,
        log: bool = True,
        round_ndigits: int = 2,
        data_root: Path = DATA_DIR,
    ):
        self.comp = comp
        self.nwindows = int(nwindows)
        self.norb = int(norb)
        self.nbins = int(nbins)
        self.log = bool(log)
        self.round_ndigits = int(round_ndigits)
        self.data_root = Path(data_root)

        self._train: Optional[Dict[Tuple[float, float], np.ndarray]] = None
        self._test: Optional[Dict[Tuple[float, float], np.ndarray]] = None
        self._merged: Optional[Dict[Tuple[float, float], np.ndarray]] = None
        self._meta: Optional[Dict[Tuple[float, float], Tuple[Source, Literal["train", "test"]]]] = None

    def _load(self) -> None:
        if self._merged is not None:
            return

        # temporarily override module-level DATA_ROOT in path builder by passing self.data_root
        def pkl(split: Literal["train", "test"]) -> Path:
            name = f"TS_{split}_{self.comp}_nwindows{self.nwindows}_norb{self.norb}_nbinsPb{self.nbins}"
            name += "_log.pkl" if self.log else ".pkl"
            return self.data_root / name

        def load_one(split: Literal["train", "test"]) -> Dict[Tuple[float, float], np.ndarray]:
            fp = pkl(split)
            if not fp.exists():
                raise FileNotFoundError(f"Dataset not found: {fp}")
            with fp.open("rb") as f:
                obj = dill.load(f)

            d: Dict[Tuple[float, float], np.ndarray] = {}
            for i in range(len(obj)):
                eb, qb = obj.X[i].tolist()
                key = _round_key(eb, qb, self.round_ndigits)
                y = obj.y[i].detach().cpu().numpy() if torch.is_tensor(obj.y[i]) else np.asarray(obj.y[i])
                d[key] = (10 ** y) if self.log else y
            return d

        train = load_one("train")
        test = load_one("test")

        merged: Dict[Tuple[float, float], np.ndarray] = {}
        meta: Dict[Tuple[float, float], Tuple[Source, Literal["train", "test"]]] = {}

        keys = set(train.keys()) | set(test.keys())
        for k in keys:
            in_train = k in train
            in_test = k in test

            if in_train and in_test:
                src: Source = "both"
                chosen = "test"
            elif in_test:
                src = "test"
                chosen = "test"
            else:
                src = "train"
                chosen = "train"

            series = test[k] if chosen == "test" else train[k]
            merged[k] = series
            meta[k] = (src, chosen)

        self._train = train
        self._test = test
        self._merged = merged
        self._meta = meta

    def keys(self):
        self._load()
        return self._merged.keys()  # type: ignore[union-attr]

    def get(self, eb: float, qb: float) -> SeriesWithMeta:
        self._load()
        key = _round_key(eb, qb, self.round_ndigits)
        if key not in self._merged:  # type: ignore[operator]
            raise KeyError(f"(eb,qb)={key} not found in merged train+test.")
        src, chosen = self._meta[key]  # type: ignore[index]
        return SeriesWithMeta(series=self._merged[key], source=src, chosen=chosen)  # type: ignore[index]

    def get_array(self, eb: float, qb: float) -> np.ndarray:
        """Convenience: return just the series array (no metadata)."""
        return self.get(eb, qb).series

    def loader(
        self
    ) -> Dict[Tuple[float, float], np.ndarray]:
        """Load merged (train+test) dataset."""
        self._load()

        keys = list(self._merged.keys())  # type: ignore[union-attr]
        X = np.array(keys, dtype=np.float32)
        y = np.array([np.asarray(self._merged[k], dtype=np.float32) for k in keys], dtype=np.float32)  # type: ignore[index]
        
        DSet = {}
        for i in range(len(X)):
            eb_val, qb_val = X[i]
            DSet[(round(eb_val, 2), round(qb_val, 2))] = y[i].tolist()

        return DSet