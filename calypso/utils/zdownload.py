from pathlib import Path
import hashlib
import shutil
import sys, os, json
import time 
from urllib.request import Request, urlopen

from .constants import BUNDLED_MANIFEST, CACHED_MANIFEST, DEFAULT_ARTIFACTS_DIR

def _ensure_cached_manifest() -> Path:
    CACHED_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    #if cached manifest does not exist, copy from bundled
    if not CACHED_MANIFEST.exists():
        shutil.copy2(BUNDLED_MANIFEST, CACHED_MANIFEST)
    #if cached manifest exists, but is older than bundled, update it
    elif BUNDLED_MANIFEST.stat().st_mtime > CACHED_MANIFEST.stat().st_mtime:
        shutil.copy2(BUNDLED_MANIFEST, CACHED_MANIFEST)
    else:
        print("Cached manifest is up to date.")
    return CACHED_MANIFEST

def load_manifest() -> dict:
    mp = _ensure_cached_manifest()
    with open(mp, "r") as f:
        return json.load(f)

def save_manifest(manifest: dict) -> None:
    mp = _ensure_cached_manifest()
    with open(mp, "w") as f:
        json.dump(manifest, f, indent=2)

def _default_user_dir() -> Path:
    return Path.home() / ".cache" / "calypso"

def _artifacts_dir() -> Path:
    #check if calypso artifacts directory is set by environment variable
    override = os.environ.get("CALYPSO_ARTIFACTS_DIR") or os.environ.get("CALYPSO_WEIGHTS_DIR")
    if override:
        d = Path(override).expanduser()
        d.mkdir(parents=True, exist_ok=True)
        return d

    #by default, use the package directory 
    pkg = Path(DEFAULT_ARTIFACTS_DIR)
        
    try:
        pkg.mkdir(parents=True, exist_ok=True)
        # quick writability test
        test = pkg / ".write_test"
        test.write_text("ok")
        test.unlink()
        return pkg
    except Exception as e:
        print("Write test: FAILED ->", repr(e))
        print("Falling back to user cache directory for artifacts")
        d = _default_user_dir() / "artifacts"
        d.mkdir(parents=True, exist_ok=True)
        return d


def _md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()



def _download(url: str, dst: Path, chunk_size: int = 1024 * 1024) -> None:
    tmp = dst.with_suffix(dst.suffix + ".part")

    # Some servers behave better with a User-Agent
    req = Request(url, headers={"User-Agent": "calypso/1.0"})
    downloaded = 0
    start = time.time()
    last_print = 0.0

    try:
        with urlopen(req) as r, open(tmp, "wb") as out:
            total = r.headers.get("Content-Length")
            total = int(total) if total is not None else None

            while True:
                chunk = r.read(chunk_size)
                if not chunk:
                    break

                out.write(chunk)
                downloaded += len(chunk)

                # Throttle UI updates (avoid spamming stdout)
                now = time.time()
                if now - last_print >= 0.1:
                    last_print = now

                    elapsed = max(now - start, 1e-9)
                    speed = downloaded / elapsed  # bytes/sec

                    if total:
                        frac = min(downloaded / total, 1.0)
                        bar_width = 30
                        filled = int(bar_width * frac)
                        bar = "=" * filled + "-" * (bar_width - filled)

                        pct = frac * 100
                        msg = (
                            f"\rDownloading {dst.name} "
                            f"[{bar}] {pct:6.2f}%  "
                            f"{downloaded/1e6:8.2f}/{total/1e6:.2f} MB  "
                            f"{speed/1e6:6.2f} MB/s"
                        )
                    else:
                        msg = (
                            f"\rDownloading {dst.name} "
                            f"{downloaded/1e6:8.2f} MB  "
                            f"{speed/1e6:6.2f} MB/s"
                        )

                    sys.stdout.write(msg)
                    sys.stdout.flush()

        # Ensure the progress line ends cleanly
        sys.stdout.write("\n")
        sys.stdout.flush()

        # Atomic finalize
        tmp.replace(dst)

    except Exception:
        # Optional cleanup so failed downloads don't linger
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        raise


def download_artifacts(force: bool = False) -> dict[str, Path]:
    """
    Download CALYPSO runtime artifacts from Zenodo if not already cached.

    Returns
    -------
    dict[str, Path]
        Mapping from filename to local cached path.
    """
    dst_dir = _artifacts_dir()
        
    out: dict[str, Path] = {}
    
    manifest = load_manifest()

    files = manifest["files"]
    
    for name, meta in files.items():
        dst = dst_dir / name
        url = meta["download_url"]
        expected = meta["checksum"]
        
        needs_download = (
            force
            or not dst.exists()
            or _md5(dst) != expected
        )

        if needs_download:
            print(f"[calypso] Downloading {name} ({meta['size']/1e6:.1f} MB)")
            _download(url, dst)

            got = _md5(dst)
            if got != expected:
                raise RuntimeError(
                    f"Checksum mismatch for {name}: got {got}, expected {expected}"
                )
        
        manifest['files'][name]['local_path'] = str(dst)
        save_manifest(manifest)

        out[name] = dst

    return out


def download_models(force: bool = False) -> dict[str, Path]:
    return download_artifacts(force=force)
