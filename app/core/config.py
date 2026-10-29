"""Environment configuration with portable defaults and validated session paths."""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VIDEOS_DIRECTORY = PROJECT_ROOT / "videosofkicks"


@dataclass(frozen=True)
class Settings:
    device: str = "auto"
    ball_model: str = "yolov8x.pt"
    pose_model: str = "yolov8s-pose.pt"
    model_dir: str | None = None
    allow_model_download: bool = True
    max_upload_bytes: int = 2 * 1024**3
    max_pending_jobs: int = 2
    keep_source_video: bool = False
    cloud_uploads: bool = True

    def __post_init__(self):
        if self.max_upload_bytes <= 0:
            raise ValueError("FOOTBALL_MAX_UPLOAD_BYTES must be positive")
        if self.max_pending_jobs <= 0:
            raise ValueError("FOOTBALL_MAX_PENDING_JOBS must be positive")


def get_settings():
    return Settings(
        device=os.getenv("FOOTBALL_DEVICE", "auto"),
        ball_model=os.getenv("FOOTBALL_BALL_MODEL", "yolov8x.pt"),
        pose_model=os.getenv("FOOTBALL_POSE_MODEL", "yolov8s-pose.pt"),
        model_dir=os.getenv("FOOTBALL_MODEL_DIR"),
        allow_model_download=os.getenv("FOOTBALL_ALLOW_MODEL_DOWNLOAD", "1") == "1",
        max_upload_bytes=int(os.getenv("FOOTBALL_MAX_UPLOAD_BYTES", str(2 * 1024**3))),
        max_pending_jobs=int(os.getenv("FOOTBALL_MAX_PENDING_JOBS", "2")),
        keep_source_video=os.getenv("FOOTBALL_KEEP_SOURCE_VIDEO", "0") == "1",
        cloud_uploads=os.getenv("FOOTBALL_CLOUD_UPLOADS", "1") == "1",
    )


def session_path(session_id):
    if not re.fullmatch(r"session_[0-9a-f]{8,32}", session_id):
        raise ValueError("Invalid session identifier")
    return VIDEOS_DIRECTORY / session_id


def get_session_dirs(session_id):
    directory = session_path(session_id)
    clips = directory / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    return directory, clips
