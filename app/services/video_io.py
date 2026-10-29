"""Checked video readers and H.264 encoder lifecycle."""

import math
import os
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
import cv2
from .media_probe import probe_video


def finish_encoder(process, timeout=30):
    """Even a buffered broken pipe must still drain errors and reap its child."""
    close_error = None
    try:
        process.stdin.close()
    except (BrokenPipeError, OSError) as exc:
        close_error = exc
    closed_input = process.stdin
    try:
        # communicate flushes stdin, so detach the stream already closed above.
        process.stdin = None
        try:
            _, error = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            _, error = process.communicate(timeout=5)
            raise RuntimeError("Preview encoder timed out: " + error.decode()[-1000:])
        code = process.returncode
        error = error.decode()
    finally:
        process.stdin = closed_input
        process.stderr.close()
    if code or close_error:
        raise RuntimeError(
            f"Preview encoder failed: {error[-1000:]} {close_error or ''}"
        )


def check_cancel(cancel_event):
    if cancel_event is not None and cancel_event.is_set():
        raise InterruptedError("Video processing cancelled")


@contextmanager
def open_video(path):
    probed = probe_video(path)
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("Cannot open video")
        values = [
            capture.get(key)
            for key in (
                cv2.CAP_PROP_FPS,
                cv2.CAP_PROP_FRAME_WIDTH,
                cv2.CAP_PROP_FRAME_HEIGHT,
            )
        ]
        fps, width, height = values
        if not all(math.isfinite(v) and v > 0 for v in (fps, width, height)):
            raise ValueError(
                "Missing or invalid video frame rate or dimensions"
            )
        yield (
            capture,
            {
                "fps": fps,
                "total_frames": probed['total_frames'],
                "width": int(width),
                "height": int(height),
            },
        )
    finally:
        capture.release()


def decoded_frames(capture, metadata, cancel_event=None):
    number = 0
    previous_t = -1.0
    while True:
        check_cancel(cancel_event)
        ok, frame = capture.read()
        if not ok:
            break
        raw_t = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000
        t = (
            raw_t
            if math.isfinite(raw_t) and raw_t > previous_t
            else number / metadata["fps"]
        )
        if t <= previous_t:
            t = previous_t + 1 / metadata["fps"]
        previous_t = t
        yield number, t, frame
        number += 1
    if number <= 0 or number != metadata["total_frames"]:
        raise ValueError(
            f"Incomplete video decode: {number}/{metadata['total_frames']} frames"
        )


def validate_video(path):
    """Inspect the container and verify this machine can decode its first frame."""
    with open_video(path) as (capture, metadata):
        ok, _ = capture.read()
        if not ok:
            raise ValueError('Cannot decode the first video frame with this codec')
        return metadata


class VideoEncoder:
    """A failed/cancelled encoder never replaces the final named artifact."""

    def __init__(self, path, width, height, fps):
        if (
            width <= 0
            or height <= 0
            or width % 2
            or height % 2
            or not math.isfinite(fps)
            or fps <= 0
        ):
            raise ValueError(
                "H.264 requires positive even dimensions and a finite frame rate"
            )
        self.path = Path(path)
        self.width, self.height = width, height
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(
            prefix="." + self.path.stem + ".",
            suffix=".partial.mp4",
            dir=self.path.parent,
        )
        os.close(fd)
        self.temporary = Path(name)
        self.frames = 0
        try:
            self.process = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "bgr24",
                    "-s",
                    f"{width}x{height}",
                    "-r",
                    str(fps),
                    "-i",
                    "-",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "26",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(self.temporary),
                ],
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except BaseException:
            self.temporary.unlink(missing_ok=True)
            raise

    def __enter__(self):
        return self

    def write(self, frame):
        if frame.shape != (self.height, self.width, 3):
            raise ValueError("Video frame dimensions do not match encoder")
        self.process.stdin.write(frame.tobytes())
        self.frames += 1

    def abort(self):
        if self.process.poll() is None:
            self.process.kill()
        try:
            finish_encoder(self.process)
        except (RuntimeError, OSError):
            pass
        self.temporary.unlink(missing_ok=True)

    def __exit__(self, kind, error, traceback):
        if kind is not None:
            self.abort()
            return False
        try:
            finish_encoder(self.process)
            if self.frames == 0:
                raise ValueError("Encoder produced no frames")
            with open_video(self.temporary) as (_, metadata):
                if metadata["total_frames"] != self.frames:
                    raise ValueError("Encoded frame count differs from supplied frames")
            os.replace(self.temporary, self.path)
        finally:
            self.temporary.unlink(missing_ok=True)
