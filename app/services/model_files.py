"""Persistent local weights, with staged download and publication under a lock."""
import os
import shutil
import tempfile
from pathlib import Path
from filelock import FileLock

OFFICIAL_MODEL_FILES = {'yolov8x.pt', 'yolov8s-pose.pt'}


def _download_weight(path):
    from ultralytics.utils.downloads import attempt_download_asset
    attempt_download_asset(str(path))


def ensure_model_file(model, *, model_dir=None, allow_download=True):
    """Resolve nonempty weights; YOLO validates their contents when loading."""
    source = Path(model).expanduser()
    explicit = source.is_absolute() or source.parent != Path('.')
    if explicit or source.name not in OFFICIAL_MODEL_FILES:
        if source.is_file() and source.stat().st_size:
            return source.resolve()
        raise FileNotFoundError(f'Model file is missing or empty: {source}')
    default_cache = Path(os.getenv('XDG_CACHE_HOME', Path.home()/'.cache'))/'football-kick'/'models'
    chosen_cache = model_dir or os.getenv('FOOTBALL_MODEL_DIR')
    directory = Path(chosen_cache or default_cache).expanduser().resolve()
    target = directory/source.name
    if target.is_file() and target.stat().st_size:
        return target
    # Existing weights in the working directory are imported once for backwards
    # compatibility. An explicitly chosen cache never silently selects that file.
    existing = source if not chosen_cache and source.is_file() and source.stat().st_size else None
    if not allow_download and existing is None:
        raise FileNotFoundError(f'Model file {target} is missing; preload weights before starting the server')
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(target)+'.lock', timeout=180):
        if target.is_file() and target.stat().st_size:
            return target
        with tempfile.TemporaryDirectory(prefix='.download-', dir=directory) as staging:
            pending = Path(staging)/source.name
            if existing is not None:
                shutil.copyfile(existing, pending)
            else:
                _download_weight(pending)
            if not pending.is_file() or pending.stat().st_size == 0:
                raise RuntimeError(f'Model download did not produce a complete file: {source.name}')
            with pending.open('rb') as stream:
                os.fsync(stream.fileno())
            os.replace(pending, target)
    return target
