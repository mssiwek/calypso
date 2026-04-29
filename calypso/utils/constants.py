from pathlib import Path
import numpy as np

# BINARY CONSTANTS
PB = 2 * np.pi
# MODEL CONSTANTS
norb = 10
nbins = 100

PKG_DIR = Path(__file__).resolve().parents[1]          # .../calypso/calypso
ASSETS_DIR = PKG_DIR / "assets"                        # bundled in the wheel
CACHE_DIR = Path.home() / ".cache" / "calypso"         # runtime downloads

BUNDLED_MANIFEST = ASSETS_DIR / "zenodo_manifest.json"
CACHED_MANIFEST  = CACHE_DIR / "assets" / "zenodo_manifest.json"

DEFAULT_ARTIFACTS_DIR = CACHE_DIR / "artifacts"
DEFAULT_WEIGHTS_DIR = DEFAULT_ARTIFACTS_DIR
DEFAULT_DATA_DIR = CACHE_DIR / "TS_data"
DEFAULT_RUNTIME_META = ASSETS_DIR / "runtime.json"
