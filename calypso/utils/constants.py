from pathlib import Path
import numpy as np

# BINARY CONSTANTS
PB = 2 * np.pi
# MODEL CONSTANTS
norb = 10
nbins = 100

PKG_DIR = Path(__file__).resolve().parents[1]          # .../calypso/calypso
REPO_ARTIFACTS_DIR = PKG_DIR.parent / ".calypso"     # pre-install only
CACHE_DIR = Path.home() / ".cache" / "calypso"       # runtime downloads

BUNDLED_MANIFEST = REPO_ARTIFACTS_DIR / "assets" / "zenodo_manifest.json"
CACHED_MANIFEST  = CACHE_DIR / "assets" / "zenodo_manifest.json"

DEFAULT_WEIGHTS_DIR = CACHE_DIR / "weights"            # keep override logic in download.py
DEFAULT_DATA_DIR = PKG_DIR.parent / ".calypso" / "TS_data"  # pre-install only
DEFAULT_MDATA_DIR = PKG_DIR.parent / ".calypso" / "assets" / "models_meta.json"  # pre-install only

TMOD_PATH = PKG_DIR / "models" / "models.py"