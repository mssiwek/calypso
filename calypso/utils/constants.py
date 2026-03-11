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

DEFAULT_ARTIFACTS_DIR = CACHE_DIR / "artifacts"
DEFAULT_WEIGHTS_DIR = DEFAULT_ARTIFACTS_DIR
BUNDLED_DATA_DIR = REPO_ARTIFACTS_DIR / "TS_data"           # pre-install only
DEFAULT_DATA_DIR = CACHE_DIR / "TS_data"                    # runtime downloads
DEFAULT_MDATA_DIR = PKG_DIR.parent / ".calypso" / "assets" / "models_meta.json"  # pre-install only
DEFAULT_RUNTIME_META = REPO_ARTIFACTS_DIR / "assets" / "runtime.json"
