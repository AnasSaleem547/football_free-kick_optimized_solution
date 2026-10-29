"""Bounded upload copying and basic container/decode validation."""

import shutil
import uuid
from pathlib import Path
from fastapi import HTTPException, UploadFile
from app.core.config import get_session_dirs, get_settings
from .video_io import validate_video
from .media_probe import VIDEO_CONTAINER_FAMILIES

ALLOWED_VIDEO_EXTENSIONS = set(VIDEO_CONTAINER_FAMILIES)
CHUNK_BYTES = 1024 * 1024


def _sanitize_video_name(stem):
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)
    return safe.strip("_")[:120] or "video"


def validate_extension(filename):
    if Path(filename).suffix.lower() not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(400, "Unsupported video extension")


def save_uploaded_file(file: UploadFile):
    if not file.filename:
        raise HTTPException(400, "No filename provided")
    validate_extension(file.filename)
    limit = get_settings().max_upload_bytes
    session_id = "session_" + uuid.uuid4().hex
    directory, clips = get_session_dirs(session_id)
    try:
        file.file.seek(0)
        path = directory / ("input" + Path(file.filename).suffix.lower())
        count = 0
        with path.open("wb") as output:
            while chunk := file.file.read(CHUNK_BYTES):
                count += len(chunk)
                if count > limit:
                    raise HTTPException(
                        413, "Video exceeds configured upload size limit"
                    )
                output.write(chunk)
        validate_video(path)
        return (
            session_id,
            str(path),
            str(clips),
            _sanitize_video_name(Path(file.filename).stem),
        )
    except BaseException as error:
        shutil.rmtree(directory, ignore_errors=True)
        if isinstance(error, ValueError):
            raise HTTPException(400, str(error)) from error
        raise
