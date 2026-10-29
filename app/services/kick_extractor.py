"""Timestamp-based annotated kick clip windows."""

from pathlib import Path
from .video_io import VideoEncoder


def clip_windows(events, duration, before=2.0, after=3.0):
    if duration <= 0 or before < 0 or after <= 0:
        raise ValueError("Invalid clip context or video duration")
    return [
        {
            "kick_number": i + 1,
            "ball_id": e.get("ball_id"),
            "kick_time_s": float(e["t"]),
            "start_s": max(0.0, float(e["t"]) - before),
            "end_s": min(float(duration), float(e["t"]) + after),
        }
        for i, e in enumerate(events)
    ]


class ClipExtractor:
    """Write independent, possibly overlapping windows during one render pass."""

    def __init__(self, output_dir, windows, width, height, fps):
        self.output_dir = Path(output_dir)
        self.windows = windows
        self.width, self.height, self.fps = width, height, fps
        self.active = {}
        self.clips = []
        self.closed = set()

    def __enter__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self

    def write(self, t, frame):
        for window in self.windows:
            key = window["kick_number"]
            if key in self.closed:
                continue
            if key in self.active and t >= window["end_s"]:
                self.active.pop(key).__exit__(None, None, None)
                self.closed.add(key)
            if window["start_s"] <= t < window["end_s"]:
                if key not in self.active:
                    path = self.output_dir / f"kick_{key:03d}.mp4"
                    encoder = VideoEncoder(path, self.width, self.height, self.fps)
                    self.active[key] = encoder
                    self.clips.append(
                        {
                            "kick_number": key,
                            "kick_start_time": round(window["kick_time_s"], 3),
                            "ball_id": window.get("ball_id"),
                            "file_path": str(path),
                            "cloud_url": None,
                            "clip_start_time": window["start_s"],
                            "clip_end_time": window["end_s"],
                        }
                    )
                self.active[key].write(frame)

    def __exit__(self, kind, error, traceback):
        try:
            if kind is None:
                for key in list(self.active):
                    self.active.pop(key).__exit__(None, None, None)
                return False
        except BaseException:
            self._discard()
            raise
        self._discard()
        return False

    def _discard(self):
        for encoder in self.active.values():
            encoder.abort()
        self.active.clear()
        for clip in self.clips:
            Path(clip["file_path"]).unlink(missing_ok=True)
