"""Lazy shared model weights, with synchronized inference and no shared tracking state."""

import threading
from pathlib import Path
from .ball_detector import BallDetector
from .pose_detector import PoseDetector
from .model_files import ensure_model_file

_RUNTIME_LOCK = threading.Lock()
_RUNTIME = None


def resolve_device(requested, cuda_available):
    if requested not in ("auto", "cpu", "cuda", "0"):
        raise ValueError("FOOTBALL_DEVICE must be auto, cpu or cuda")
    if requested in ("cuda", "0") and not cuda_available:
        raise ValueError("CUDA requested, but no CUDA device is available")
    cuda = cuda_available and requested != "cpu"
    return (0, True) if cuda else ("cpu", False)


class ModelRuntime:
    def __init__(
        self, ball_path="yolov8x.pt", pose_path="yolov8s-pose.pt", device="auto",
        *, model_dir=None, allow_download=True,
    ):
        import torch
        import ultralytics
        from ultralytics import YOLO
        from .evidence import sha256_file

        self.device, self.half = resolve_device(device, torch.cuda.is_available())
        ball_file = ensure_model_file(ball_path, model_dir=model_dir, allow_download=allow_download)
        pose_file = ensure_model_file(pose_path, model_dir=model_dir, allow_download=allow_download)
        self.ball = BallDetector(YOLO(str(ball_file)))
        self.pose = PoseDetector(YOLO(str(pose_file)))
        self.lock = threading.Lock()
        self.fingerprint = {
            "models": {
                ball_path: sha256_file(Path(self.ball.model.ckpt_path)),
                pose_path: sha256_file(Path(self.pose.model.ckpt_path)),
            },
            "package_version": ultralytics.__version__,
            "device": self.device,
            "half": self.half,
        }

    def predict(self, frames, pose_indices, settings):
        with self.lock:
            balls = self.ball.predict(
                frames,
                device=self.device,
                half=self.half,
                imgsz=settings["ball_imgsz"],
                conf=settings["ball_conf"],
            )
            poses = self.pose.predict(
                [frames[i] for i in pose_indices],
                device=self.device,
                half=self.half,
                imgsz=settings["pose_imgsz"],
            )
        return balls, poses


def get_runtime():
    global _RUNTIME
    from app.core.config import get_settings

    cfg = get_settings()
    with _RUNTIME_LOCK:
        if _RUNTIME is None:
            _RUNTIME = ModelRuntime(
                cfg.ball_model, cfg.pose_model, cfg.device,
                model_dir=cfg.model_dir, allow_download=cfg.allow_model_download,
            )
        return _RUNTIME
