from __future__ import annotations

import json
from pathlib import Path

from .reconstruction.emulator import PCAEmulator
from .utils.constants import DEFAULT_RUNTIME_META
from .utils.zdownload import download_files, load_manifest


def _default_artifact_name() -> str:
    if DEFAULT_RUNTIME_META.exists():
        with open(DEFAULT_RUNTIME_META, "r") as f:
            meta = json.load(f)
        name = meta.get("runtime_artifact")
        if isinstance(name, str) and name:
            return name

    manifest = load_manifest()
    explicit = manifest.get("runtime_artifact")
    if isinstance(explicit, str) and explicit:
        return explicit

    files = manifest.get("files", {})
    if len(files) == 1:
        return next(iter(files))

    raise RuntimeError(
        "No default runtime artifact configured. "
        "Set `runtime_artifact` in runtime.json or zenodo_manifest.json."
    )


def load_emulator(artifact_name: str | None = None, force_download: bool = False) -> PCAEmulator:
    selected = artifact_name or _default_artifact_name()
    artifacts = download_files([selected], force=force_download)
    path = artifacts.get(selected)
    if path is None:
        raise FileNotFoundError(f"Runtime artifact not found in manifest: {selected}")
    return PCAEmulator.load(Path(path))
