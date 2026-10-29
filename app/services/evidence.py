"""Model evidence cache and extraction."""

import gzip
import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path
import cv2
import numpy as np
from .camera import camera_translation
from .video_io import open_video, decoded_frames, check_cancel


def cache_matches(evidence, sha256, settings, model_hashes, revision, package_version):
    return (
        isinstance(evidence, dict)
        and evidence.get("sha256") == sha256
        and evidence.get("settings") == settings
        and evidence.get("model_sha256") == model_hashes
        and evidence.get("extractor_revision") == revision
        and evidence.get("package_version") == package_version
    )


INFERENCE_SETTINGS = {
    "ball_imgsz": 1280,
    "pose_imgsz": 960,
    "ball_conf": 0.12,
    "pose_stride": 2,
    "batch": 6,
}
EXTRACTOR_REVISION = (
    3  # Runtime/device provenance and checked decode/cache publication.
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _complete_cache(record):
    """Validate cached structure before passing it to analysis or rendering."""
    def vector(value, size):
        return (isinstance(value, list) and len(value) == size
                and all(isinstance(v, (int, float)) and math.isfinite(v) for v in value))

    try:
        frames = record["frames"]
        if (not isinstance(frames, list) or not frames
                or len(frames) != record["decoded_frames"]
                or len(frames) != record["total_frames"]
                or not math.isfinite(record["fps"]) or record["fps"] <= 0):
            return False
        previous_t = -math.inf
        for index, frame in enumerate(frames):
            if (not isinstance(frame, dict) or frame.get("frame") != index
                    or not math.isfinite(frame["t"]) or frame["t"] <= previous_t
                    or not vector(frame["shift"], 2)
                    or not isinstance(frame["balls"], list)
                    or not all(vector(ball, 4) and ball[2] > 0 and 0 <= ball[3] <= 1
                               for ball in frame["balls"])
                    or not isinstance(frame["poses"], list)):
                return False
            for pose in frame["poses"]:
                if (not vector(pose["box"], 4) or not vector(pose["hip"], 2)
                        or not isinstance(pose["ankles"], list)
                        or not all(vector(foot, 2) for foot in pose["ankles"])):
                    return False
            previous_t = frame["t"]
        return True
    except (KeyError, TypeError):
        return False


def extract_evidence(
    video_path,
    work_dir,
    *,
    runtime=None,
    settings=None,
    cancel_event=None,
    progress=None,
):
    check_cancel(cancel_event)
    if runtime is None:
        from .model_runtime import get_runtime

        runtime = get_runtime()
    cfg = dict(INFERENCE_SETTINGS)
    cfg.update(settings or {})
    if cfg["batch"] < 1 or cfg["pose_stride"] < 1:
        raise ValueError("Batch size and pose stride must be positive")
    fingerprint = runtime.fingerprint
    cfg.update(device=fingerprint["device"], half=fingerprint["half"])
    source_sha = sha256_file(video_path)
    work_dir = Path(work_dir)
    cache = work_dir / (source_sha + ".evidence.json.gz")
    if cache.exists():
        try:
            with gzip.open(cache, "rt") as stream:
                old = json.load(stream)
            if (
                cache_matches(
                    old,
                    source_sha,
                    cfg,
                    fingerprint["models"],
                    EXTRACTOR_REVISION,
                    fingerprint["package_version"],
                )
                and _complete_cache(old)
            ):
                check_cancel(cancel_event)
                return old
        except (ValueError, KeyError, OSError, EOFError):
            pass  # A corrupt/interrupted cache never substitutes for source decoding.
    frames = []
    batch = []
    records = []
    cumulative = np.zeros(2)
    previous = None
    started = time.perf_counter()
    inference_s = 0.0
    batch_limit = cfg["batch"] if fingerprint["device"] != "cpu" else 1

    def flush():
        nonlocal inference_s
        if not batch:
            return
        check_cancel(cancel_event)
        indices = [
            i for i, r in enumerate(records) if r["frame"] % cfg["pose_stride"] == 0
        ]
        tick = time.perf_counter()
        balls, poses = runtime.predict(batch, indices, cfg)
        if len(balls) != len(records) or len(poses) != len(indices):
            raise RuntimeError("Model output batch is incomplete")
        inference_s += time.perf_counter() - tick
        for record, value in zip(records, balls):
            record["balls"] = value
        for index, value in zip(indices, poses):
            records[index]["poses"] = value
        frames.extend(records)
        records.clear()
        batch.clear()
        if progress:
            progress(len(frames), metadata["total_frames"])

    with open_video(video_path) as (capture, metadata):
        for number, t, frame in decoded_frames(capture, metadata, cancel_event):
            h, w = frame.shape[:2]
            scale = 640.0 / w
            gray = cv2.cvtColor(
                cv2.resize(frame, (640, round(h * scale))), cv2.COLOR_BGR2GRAY
            )
            reliable = True
            cut = False
            if previous is not None:
                mask = np.zeros_like(gray)
                mask[: int(0.60 * len(gray))] = 255
                points = cv2.goodFeaturesToTrack(previous, 160, 0.02, 8, mask=mask)
                reliable = False
                if points is not None:
                    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                        previous, gray, points, None
                    )
                    if next_points is not None:
                        keep = status.reshape(-1) == 1
                        shift, valid = camera_translation(
                            points.reshape(-1, 2)[keep],
                            next_points.reshape(-1, 2)[keep],
                        )
                        if valid and np.linalg.norm(shift) < 20:
                            cumulative += shift / scale
                            reliable = True
                        elif valid:
                            cut = True
            previous = gray
            records.append(
                {
                    "frame": number,
                    "t": t,
                    "width": w,
                    "height": h,
                    "balls": [],
                    "poses": [],
                    "shift": list(map(float, cumulative)),
                    "camera_cut": cut,
                    "camera_reliable": reliable,
                }
            )
            batch.append(frame)
            if len(batch) >= batch_limit:
                flush()
        flush()
    check_cancel(cancel_event)
    evidence = {
        "name": Path(video_path).name,
        "sha256": source_sha,
        "fps": metadata["fps"],
        "total_frames": metadata["total_frames"],
        "decoded_frames": len(frames),
        "frames": frames,
        "model_sha256": fingerprint["models"],
        "settings": cfg,
        "extractor_revision": EXTRACTOR_REVISION,
        "package_version": fingerprint["package_version"],
        "inference_s": inference_s,
        "decode_and_inference_s": time.perf_counter() - started,
    }
    work_dir.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".evidence.", suffix=".tmp", dir=work_dir)
    os.close(fd)
    try:
        with gzip.open(name, "wt") as stream:
            json.dump(evidence, stream, allow_nan=False)
        os.replace(name, cache)
    finally:
        Path(name).unlink(missing_ok=True)
    return evidence
